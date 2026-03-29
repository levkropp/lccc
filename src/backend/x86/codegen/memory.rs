//! X86Codegen: memory operations (load, store, memcpy, GEP, stack).

use crate::ir::reexports::{IrConst, IrBinOp, Instruction, Operand, Value};
use crate::common::types::{AddressSpace, IrType};
use crate::backend::state::{StackSlot, SlotAddr};
use super::emit::{X86Codegen, phys_reg_name, typed_phys_reg_name, is_xmm_reg};

impl X86Codegen {
    /// Try to emit a store using x86-64 SIB indexed addressing mode.
    /// Returns true if successful, false to fall back to normal codegen.
    ///
    /// First tries IVSR pattern detection (Phase 9b), then falls back to
    /// Phase 9 pattern (GEP with Mul/Shl offset).
    fn try_emit_indexed_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) -> bool {
        // Phase 9b: Try IVSR pointer pattern first (most common in loops)
        if self.try_emit_ivsr_indexed_store(val, ptr, ty) {
            return true;
        }

        // Phase 9: Try non-IVSR pattern (explicit multiply/shift)
        self.try_emit_phase9_indexed_store(val, ptr, ty)
    }

    /// Try to emit indexed addressing for IVSR-transformed loop pointer stores.
    fn try_emit_ivsr_indexed_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) -> bool {
        // Check if ptr is an IVSR pointer phi
        let ivsr_info = match self.ivsr_pointers.get(&ptr.0) {
            Some(info) => info.clone(),
            None => return false,
        };

        // Find the loop counter associated with this pointer
        let counter_val = match self.pointer_to_counter.get(&ptr.0) {
            Some(&counter_id) => Value(counter_id),
            None => return false,
        };

        // Check if both base and counter are in registers
        let base_reg = match self.reg_assignments.get(&ivsr_info.base_ptr.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        let index_reg = match self.reg_assignments.get(&counter_val.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        // Verify stride is a valid SIB scale
        if !Self::is_valid_sib_scale(ivsr_info.stride) {
            return false;
        }

        // Base and index must be different registers.
        if base_reg == index_reg {
            return false;
        }

        // Check if loading the store value would clobber base or index register
        if let Operand::Value(v) = val {
            if let Some(&val_reg) = self.reg_assignments.get(&v.0) {
                let val_name = phys_reg_name(val_reg);
                if val_name == base_reg || val_name == index_reg {
                    return false;
                }
            }
        }

        // Load the value to be stored into the accumulator/xmm register
        self.operand_to_rax(val);

        // Determine store instruction and source register based on type
        let (store_instr, src_reg) = match ty {
            IrType::F64 => {
                // Convert from rax to xmm0
                self.state.emit("    movq %rax, %xmm0");
                ("movsd", "%xmm0")
            }
            IrType::F32 => {
                // Convert from rax to xmm0
                self.state.emit("    movd %eax, %xmm0");
                ("movss", "%xmm0")
            }
            IrType::I64 | IrType::U64 => ("movq", "%rax"),
            IrType::I32 | IrType::U32 => ("movl", "%eax"),
            IrType::I16 | IrType::U16 => ("movw", "%ax"),
            IrType::I8 | IrType::U8 => ("movb", "%al"),
            _ => return false,
        };

        // Emit indexed store: movX %src, (%base,%index,scale)
        if ivsr_info.init_offset == 0 {
            self.state.emit_fmt(format_args!(
                "    {} {}, (%{},%{},{})",
                store_instr, src_reg, base_reg, index_reg, ivsr_info.stride
            ));
        } else {
            self.state.emit_fmt(format_args!(
                "    {} {}, {}(%{},%{},{})",
                store_instr, src_reg, ivsr_info.init_offset, base_reg, index_reg, ivsr_info.stride
            ));
        }

        true
    }

    /// Phase 9 indexed addressing: detect GEP with multiply/shift offset for stores.
    /// Detects patterns like: `store val, (base + index*scale)` where:
    /// - base is in a register
    /// - index is in a register
    /// - scale is 1, 2, 4, or 8
    ///
    /// Emits: `mov %src, (%base_reg,%index_reg,scale)`
    fn try_emit_phase9_indexed_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) -> bool {
        // Phase 9 decomposes a variable-offset GEP into SIB addressing:
        //   Store val, (GEP base, Mul(idx, scale))  →  movl %eax, (%base, %idx, scale)
        // However, variable-offset GEPs are always emitted as `leaq` instructions
        // (they are NOT in gep_fold_map, which only handles constant offsets).
        // By the time the Store is emitted, the GEP's source registers (base, idx)
        // may have been clobbered by the leaq destination or intervening instructions.
        // The GEP result is already computed in a register/slot, so use it directly.
        return false;

        // Check if ptr is defined by a GEP instruction
        let gep_inst = match self.get_defining_instruction(ptr.0) {
            Some(inst) => inst,
            None => return false,
        };

        let (gep_base, gep_offset) = match gep_inst {
            Instruction::GetElementPtr { base, offset, .. } => (base, offset),
            _ => return false,
        };

        // Check if offset is a Value (not a constant - those are handled by existing GEP folding)
        let offset_val = match gep_offset {
            Operand::Value(v) => v,
            _ => return false,
        };

        // Check if offset is defined by a multiply or shift (i*scale pattern)
        let (index_val, scale) = match self.get_defining_instruction(offset_val.0) {
            Some(Instruction::BinOp { op: IrBinOp::Mul, lhs: Operand::Value(idx), rhs: Operand::Const(c), .. }) => {
                // Pattern: index * const
                let scale_val = match c.to_i64() {
                    Some(v) => v,
                    None => return false,
                };
                if !Self::is_valid_sib_scale(scale_val) {
                    return false;
                }
                (idx, scale_val)
            }
            Some(Instruction::BinOp { op: IrBinOp::Shl, lhs: Operand::Value(idx), rhs: Operand::Const(c), .. }) => {
                // Pattern: index << shift_amount (equivalent to index * 2^shift)
                let shift = match c.to_i64() {
                    Some(v) if v >= 0 && v <= 3 => v,  // shift of 0-3 gives scale of 1,2,4,8
                    _ => return false,
                };
                let scale_val = 1i64 << shift;
                (idx, scale_val)
            }
            _ => return false,
        };

        // Check if base and index both have register assignments
        let base_reg = match self.reg_assignments.get(&gep_base.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        let index_reg = match self.reg_assignments.get(&index_val.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        // Base and index must be different registers for SIB addressing.
        // If the register allocator assigned both the same register (e.g.,
        // because one value's live range ended and the register was reused),
        // the SIB computation would be wrong (base + base*scale instead of
        // base + index*scale).
        if base_reg == index_reg {
            return false;
        }

        // Check if loading the store value would clobber the base or index
        // register. This happens when the store value's register overlaps with
        // the base/index, or when operand_to_rax needs to use the register for
        // intermediate computations. If so, fall back to non-indexed store.
        if let Operand::Value(v) = val {
            if let Some(&val_reg) = self.reg_assignments.get(&v.0) {
                let val_name = phys_reg_name(val_reg);
                if val_name == base_reg || val_name == index_reg {
                    return false; // Register conflict, fall back
                }
            }
        }

        // Load the value to be stored into the accumulator/xmm register
        self.operand_to_rax(val);

        // Determine store instruction and source register based on type
        let (store_instr, src_reg) = match ty {
            IrType::F64 => {
                // Convert from rax to xmm0
                self.state.emit("    movq %rax, %xmm0");
                ("movsd", "%xmm0")
            }
            IrType::F32 => {
                // Convert from rax to xmm0
                self.state.emit("    movd %eax, %xmm0");
                ("movss", "%xmm0")
            }
            IrType::I64 | IrType::U64 => ("movq", "%rax"),
            IrType::I32 | IrType::U32 => ("movl", "%eax"),
            IrType::I16 | IrType::U16 => ("movw", "%ax"),
            IrType::I8 | IrType::U8 => ("movb", "%al"),
            _ => return false,  // Unsupported type for indexed addressing
        };

        // Emit indexed store: movX %src, (%base,%index,scale)
        self.state.emit_fmt(format_args!(
            "    {} {}, (%{},%{},{})",
            store_instr, src_reg, base_reg, index_reg, scale
        ));

        true
    }

    /// Try to emit a load using x86-64 SIB indexed addressing mode.
    /// Returns true if successful, false to fall back to normal codegen.
    ///
    /// First tries IVSR pattern detection (Phase 9b), then falls back to
    /// Phase 9 pattern (GEP with Mul/Shl offset).
    fn try_emit_indexed_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) -> bool {
        // Phase 9b: Try IVSR pointer pattern first (most common in loops)
        if self.try_emit_ivsr_indexed_load(dest, ptr, ty) {
            return true;
        }

        // Phase 9: Try non-IVSR pattern (explicit multiply/shift)
        self.try_emit_phase9_indexed_load(dest, ptr, ty)
    }

    /// Try to emit indexed addressing for IVSR-transformed loop pointers.
    /// Detects pattern: %ptr = Phi(%init, %next) where %next = GEP(%ptr, stride)
    /// and emits: movX (%base,%counter,scale), %dest
    fn try_emit_ivsr_indexed_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) -> bool {
        // Check if ptr is an IVSR pointer phi
        let ivsr_info = match self.ivsr_pointers.get(&ptr.0) {
            Some(info) => info.clone(),
            None => return false,
        };

        // Find the loop counter associated with this pointer
        let counter_val = match self.pointer_to_counter.get(&ptr.0) {
            Some(&counter_id) => Value(counter_id),
            None => return false,
        };

        // Check if both base and counter are in registers
        let base_reg = match self.reg_assignments.get(&ivsr_info.base_ptr.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        let index_reg = match self.reg_assignments.get(&counter_val.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        // Verify stride is a valid SIB scale
        if !Self::is_valid_sib_scale(ivsr_info.stride) {
            return false;
        }

        // Determine load instruction and destination register based on type
        let (load_instr, dest_reg) = match ty {
            IrType::F64 => ("movsd", "%xmm0"),
            IrType::F32 => ("movss", "%xmm0"),
            IrType::I64 | IrType::U64 => ("movq", "%rax"),
            IrType::I32 | IrType::U32 => ("movl", "%eax"),
            IrType::I16 | IrType::U16 => ("movzwl", "%eax"),
            IrType::I8 | IrType::U8 => ("movzbl", "%eax"),
            _ => return false,
        };

        // Emit indexed load: movX (%base,%index,scale), %dest
        // Handle optional displacement for non-zero init_offset
        if ivsr_info.init_offset == 0 {
            self.state.emit_fmt(format_args!(
                "    {} (%{},%{},{}), {}",
                load_instr, base_reg, index_reg, ivsr_info.stride, dest_reg
            ));
        } else {
            self.state.emit_fmt(format_args!(
                "    {} {}(%{},%{},{}), {}",
                load_instr, ivsr_info.init_offset, base_reg, index_reg, ivsr_info.stride, dest_reg
            ));
        }

        // Update register cache - for FP types, value is in xmm0, for integers in rax
        match ty {
            IrType::F64 | IrType::F32 => {
                // For floating point, the value is in xmm0, not rax
                // We need to move it to rax for the common code path
                if ty == IrType::F64 {
                    self.state.emit("    movq %xmm0, %rax");
                } else {
                    self.state.emit("    movd %xmm0, %eax");
                }
                self.state.reg_cache.set_acc(dest.0, false);
            }
            _ => {
                // Integer types are already in rax
                self.state.reg_cache.set_acc(dest.0, false);
            }
        }

        true
    }

    /// Phase 9 indexed addressing: detect GEP with multiply/shift offset.
    /// Detects patterns like: `load (base + index*scale)` where:
    /// - base is in a register
    /// - index is in a register
    /// - scale is 1, 2, 4, or 8
    ///
    /// Emits: `mov (%base_reg,%index_reg,scale), %dest`
    fn try_emit_phase9_indexed_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) -> bool {
        // Disabled: same issue as try_emit_phase9_indexed_store — variable-offset
        // GEPs are already emitted, so base/index registers may be stale.
        return false;

        // Check if ptr is defined by a GEP instruction
        let gep_inst = match self.get_defining_instruction(ptr.0) {
            Some(inst) => inst,
            None => return false,
        };

        let (gep_base, gep_offset) = match gep_inst {
            Instruction::GetElementPtr { base, offset, .. } => (base, offset),
            _ => return false,
        };

        // Check if offset is a Value (not a constant - those are handled by existing GEP folding)
        let offset_val = match gep_offset {
            Operand::Value(v) => v,
            _ => return false,
        };

        // Check if offset is defined by a multiply or shift (i*scale pattern)
        let (index_val, scale) = match self.get_defining_instruction(offset_val.0) {
            Some(Instruction::BinOp { op: IrBinOp::Mul, lhs: Operand::Value(idx), rhs: Operand::Const(c), .. }) => {
                // Pattern: index * const
                let scale_val = match c.to_i64() {
                    Some(v) => v,
                    None => return false,
                };
                if !Self::is_valid_sib_scale(scale_val) {
                    return false;
                }
                (idx, scale_val)
            }
            Some(Instruction::BinOp { op: IrBinOp::Shl, lhs: Operand::Value(idx), rhs: Operand::Const(c), .. }) => {
                // Pattern: index << shift_amount (equivalent to index * 2^shift)
                let shift = match c.to_i64() {
                    Some(v) if v >= 0 && v <= 3 => v,  // shift of 0-3 gives scale of 1,2,4,8
                    _ => return false,
                };
                let scale_val = 1i64 << shift;
                (idx, scale_val)
            }
            _ => return false,
        };

        // Check if base and index both have register assignments
        let base_reg = match self.reg_assignments.get(&gep_base.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        let index_reg = match self.reg_assignments.get(&index_val.0) {
            Some(&reg) => phys_reg_name(reg),
            None => return false,
        };

        // Determine load instruction and destination register based on type
        let (load_instr, dest_reg) = match ty {
            IrType::F64 => ("movsd", "%xmm0"),
            IrType::F32 => ("movss", "%xmm0"),
            IrType::I64 | IrType::U64 => ("movq", "%rax"),
            IrType::I32 | IrType::U32 => ("movl", "%eax"),
            IrType::I16 | IrType::U16 => ("movzwl", "%eax"),
            IrType::I8 | IrType::U8 => ("movzbl", "%eax"),
            _ => return false,  // Unsupported type for indexed addressing
        };

        // Emit indexed load: movX (%base,%index,scale), %dest
        self.state.emit_fmt(format_args!(
            "    {} (%{},%{},{}), {}",
            load_instr, base_reg, index_reg, scale, dest_reg
        ));

        // Update register cache - for FP types, value is in xmm0, for integers in rax
        match ty {
            IrType::F64 | IrType::F32 => {
                // For floating point, the value is in xmm0, not rax
                // We need to move it to rax for the common code path
                if ty == IrType::F64 {
                    self.state.emit("    movq %xmm0, %rax");
                } else {
                    self.state.emit("    movd %xmm0, %eax");
                }
                self.state.reg_cache.set_acc(dest.0, false);
            }
            _ => {
                // Integer types are already in rax
                self.state.reg_cache.set_acc(dest.0, false);
            }
        }

        true
    }

    // ---- Store/Load overrides ----

    pub(super) fn emit_store_impl(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        if ty == IrType::F128 {
            if let Operand::Const(IrConst::LongDouble(_, f128_bytes)) = val {
                let x87 = crate::common::long_double::f128_bytes_to_x87_bytes(f128_bytes);
                let lo = u64::from_le_bytes(x87[0..8].try_into().unwrap());
                let hi_bytes: [u8; 8] = [x87[8], x87[9], 0, 0, 0, 0, 0, 0];
                let hi = u64::from_le_bytes(hi_bytes);
                if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                    self.emit_f128_store_raw_bytes(&addr, ptr.0, 0, lo, hi);
                }
                return;
            }
            if let Operand::Value(v) = val {
                if self.state.f128_direct_slots.contains(&v.0) {
                    if let Some(src_slot) = self.state.get_slot(v.0) {
                        if let Some(dest_addr) = self.state.resolve_slot_addr(ptr.0) {
                            self.state.out.emit_instr_rbp("    fldt", src_slot.0);
                            self.emit_f128_fstpt(&dest_addr, ptr.0, 0);
                            return;
                        }
                    }
                }
            }
            self.operand_to_rax(val);
            if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                self.emit_f128_store_f64_via_x87(&addr, ptr.0, 0);
            }
            return;
        }

        // Try indexed addressing mode first (Phase 9 optimization)
        if self.try_emit_indexed_store(val, ptr, ty) {
            // Indexed addressing succeeded - we're done!
            return;
        }

        // Constant-immediate store optimization: when storing a small constant,
        // emit `movX $IMM, ADDR` directly instead of loading the constant into
        // %rax and then storing through the accumulator. This saves 1-3 instructions
        // (eliminates xorl/movq for zero, or movq $IMM for other constants, plus
        // the emit_save_acc movq %rax, %rdx for indirect stores).
        if !ty.is_float() && !matches!(ty, IrType::I128 | IrType::U128 | IrType::F128) {
            if let Operand::Const(c) = val {
                if let Some(imm) = c.to_i64() {
                    // Only use immediate form when value fits in i32 (x86 mov mem,imm limitation)
                    if imm >= i32::MIN as i64 && imm <= i32::MAX as i64 {
                        if self.try_emit_const_store(imm as i32, ptr, ty) {
                            return;
                        }
                    }
                }
            }
        }

        // Register-direct fast path: bypass the accumulator when both val
        // and ptr have register assignments. Saves 2-3 instructions per store.
        if !ty.is_float() && !matches!(ty, IrType::I128 | IrType::U128 | IrType::F128) {
            if let Operand::Value(v) = val {
                let v_reg = self.reg_assignments.get(&v.0).copied();
                let p_reg = self.reg_assignments.get(&ptr.0).copied();

                // Both val and ptr in GPR registers: emit 1 instruction
                if let (Some(vr), Some(pr)) = (v_reg, p_reg) {
                    if !is_xmm_reg(vr) && !is_xmm_reg(pr) && !self.state.is_alloca(ptr.0) {
                        let store_instr = Self::mov_store_for_type(ty);
                        let v_name = typed_phys_reg_name(vr, ty);
                        let p_name = phys_reg_name(pr);
                        self.state.emit_fmt(format_args!("    {} %{}, (%{})", store_instr, v_name, p_name));
                        self.state.reg_cache.invalidate_acc();
                        return;
                    }
                }

                // Val in register, ptr on stack: 2 instructions (skip emit_save_acc)
                if let Some(vr) = v_reg {
                    if !is_xmm_reg(vr) && !self.state.is_alloca(ptr.0) {
                        if let Some(crate::backend::state::SlotAddr::Indirect(slot)) = self.state.resolve_slot_addr(ptr.0) {
                            self.emit_load_ptr_from_slot_impl(slot, ptr.0);
                            let store_instr = Self::mov_store_for_type(ty);
                            let v_name = typed_phys_reg_name(vr, ty);
                            self.state.emit_fmt(format_args!("    {} %{}, (%rcx)", store_instr, v_name));
                            self.state.reg_cache.invalidate_acc();
                            return;
                        }
                    }
                }
            }
        }

        // Fall back to default store logic
        crate::backend::traits::emit_store_default(self, val, ptr, ty);
    }

    /// Try to emit a constant-immediate store: `movX $IMM, ADDR`.
    /// Bypasses the accumulator entirely, saving 1-3 instructions.
    fn try_emit_const_store(&mut self, imm: i32, ptr: &Value, ty: IrType) -> bool {
        let store_instr = Self::mov_store_for_type(ty);
        let addr = self.state.resolve_slot_addr(ptr.0);

        match addr {
            Some(SlotAddr::Direct(slot)) => {
                // Store constant directly to stack slot: movX $IMM, N(%rsp/%rbp)
                let out = &mut self.state.out;
                out.write_str("    ");
                out.write_str(store_instr);
                out.write_str(" $");
                out.write_i64(imm as i64);
                out.write_str(", ");
                if out.use_rsp_addressing {
                    out.write_i64(out.rsp_frame_size + slot.0);
                    out.write_str("(%rsp)");
                } else {
                    out.write_i64(slot.0);
                    out.write_str("(%rbp)");
                }
                out.newline();
                self.state.reg_cache.invalidate_all();
                true
            }
            Some(SlotAddr::Indirect(slot)) => {
                // Pointer is in a stack slot — load pointer to %rcx, then store immediate
                self.emit_load_ptr_from_slot_impl(slot, ptr.0);
                self.state.emit_fmt(format_args!("    {} ${}, (%rcx)", store_instr, imm));
                self.state.reg_cache.invalidate_all();
                true
            }
            _ => false, // OverAligned or no slot — fall back to default
        }
    }

    pub(super) fn emit_load_impl(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if ty == IrType::F128 {
            if let Some(addr) = self.state.resolve_slot_addr(ptr.0) {
                self.emit_f128_fldt(&addr, ptr.0, 0);
                self.emit_f128_load_finish(dest);
                self.state.track_f128_load(dest.0, ptr.0, 0);
            }
            return;
        }

        // Try indexed addressing mode first (Phase 9 optimization)
        if self.try_emit_indexed_load(dest, ptr, ty) {
            // Indexed addressing succeeded - we're done!
            // The accumulator already holds dest's value (cache updated in try_emit_indexed_load)
            self.store_rax_to(dest);
            return;
        }

        // Register-direct load: when ptr has a register, load directly from
        // (%ptr_reg) into rax, skipping the rcx intermediate. Saves 1 instruction.
        if !ty.is_float() && !matches!(ty, IrType::I128 | IrType::U128 | IrType::F128) {
            if let Some(p_reg) = self.reg_assignments.get(&ptr.0).copied() {
                if !is_xmm_reg(p_reg) && !self.state.is_alloca(ptr.0) {
                    let load_instr = Self::mov_load_for_type(ty);
                    let p_name = phys_reg_name(p_reg);
                    let dest_reg = if matches!(ty, IrType::I32 | IrType::U32) { "%eax" } else { "%rax" };
                    self.state.emit_fmt(format_args!("    {} (%{}), {}", load_instr, p_name, dest_reg));
                    self.state.reg_cache.set_acc(dest.0, false);
                    self.store_rax_to(dest);
                    return;
                }
            }
        }

        // Fall back to default load logic
        crate::backend::traits::emit_load_default(self, dest, ptr, ty);
    }

    pub(super) fn emit_store_with_const_offset_impl(&mut self, val: &Operand, base: &Value, offset: i64, ty: IrType) {
        if ty == IrType::F128 {
            if let Operand::Const(IrConst::LongDouble(_, f128_bytes)) = val {
                let x87 = crate::common::long_double::f128_bytes_to_x87_bytes(f128_bytes);
                let lo = u64::from_le_bytes(x87[0..8].try_into().unwrap());
                let hi_bytes: [u8; 8] = [x87[8], x87[9], 0, 0, 0, 0, 0, 0];
                let hi = u64::from_le_bytes(hi_bytes);
                if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                    self.emit_f128_store_raw_bytes(&addr, base.0, offset, lo, hi);
                }
                return;
            }
            if let Operand::Value(v) = val {
                if self.state.f128_direct_slots.contains(&v.0) {
                    if let Some(src_slot) = self.state.get_slot(v.0) {
                        if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                            self.state.out.emit_instr_rbp("    fldt", src_slot.0);
                            self.emit_f128_fstpt(&addr, base.0, offset);
                            return;
                        }
                    }
                }
            }
            self.operand_to_rax(val);
            if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                self.emit_f128_store_f64_via_x87(&addr, base.0, offset);
            }
            return;
        }
        // Non-F128: try constant-immediate store optimization first.
        if !ty.is_float() && !matches!(ty, IrType::I128 | IrType::U128) {
            if let Operand::Const(c) = val {
                if let Some(imm) = c.to_i64() {
                    if imm >= i32::MIN as i64 && imm <= i32::MAX as i64 {
                        let store_instr = Self::mov_store_for_type(ty);
                        let imm32 = imm as i32;
                        let addr = self.state.resolve_slot_addr(base.0);
                        match addr {
                            Some(SlotAddr::Direct(slot)) => {
                                let folded_slot = StackSlot(slot.0 + offset);
                                let out = &mut self.state.out;
                                out.write_str("    ");
                                out.write_str(store_instr);
                                out.write_str(" $");
                                out.write_i64(imm32 as i64);
                                out.write_str(", ");
                                if out.use_rsp_addressing {
                                    out.write_i64(out.rsp_frame_size + folded_slot.0);
                                    out.write_str("(%rsp)");
                                } else {
                                    out.write_i64(folded_slot.0);
                                    out.write_str("(%rbp)");
                                }
                                out.newline();
                                self.state.reg_cache.invalidate_all();
                                return;
                            }
                            Some(SlotAddr::Indirect(slot)) => {
                                if let Some(&reg) = self.reg_assignments.get(&base.0) {
                                    let reg_name = phys_reg_name(reg);
                                    if offset != 0 {
                                        self.state.emit_fmt(format_args!("    {} ${}, {}(%{})", store_instr, imm32, offset, reg_name));
                                    } else {
                                        self.state.emit_fmt(format_args!("    {} ${}, (%{})", store_instr, imm32, reg_name));
                                    }
                                } else {
                                    self.emit_load_ptr_from_slot_impl(slot, base.0);
                                    if offset != 0 {
                                        self.emit_add_offset_to_addr_reg_impl(offset);
                                    }
                                    self.state.emit_fmt(format_args!("    {} ${}, (%rcx)", store_instr, imm32));
                                }
                                self.state.reg_cache.invalidate_all();
                                return;
                            }
                            _ => {} // fall through to default path
                        }
                    }
                }
            }
        }

        // Default GEP fold logic.
        self.operand_to_rax(val);
        let addr = self.state.resolve_slot_addr(base.0);
        if let Some(addr) = addr {
            let store_instr = Self::mov_store_for_type(ty);
            match addr {
                SlotAddr::OverAligned(slot, id) => {
                    self.emit_save_acc_impl();
                    self.emit_alloca_aligned_addr_impl(slot, id);
                    self.emit_add_offset_to_addr_reg_impl(offset);
                    self.emit_typed_store_indirect_impl(store_instr, ty);
                }
                SlotAddr::Direct(slot) => {
                    let folded_slot = StackSlot(slot.0 + offset);
                    self.emit_typed_store_to_slot_impl(store_instr, ty, folded_slot);
                }
                SlotAddr::Indirect(slot) => {
                    if let Some(&reg) = self.reg_assignments.get(&base.0) {
                        let reg_name = phys_reg_name(reg);
                        let store_reg = Self::reg_for_type("rax", ty);
                        if offset != 0 {
                            self.state.emit_fmt(format_args!("    {} %{}, {}(%{})", store_instr, store_reg, offset, reg_name));
                        } else {
                            self.state.emit_fmt(format_args!("    {} %{}, (%{})", store_instr, store_reg, reg_name));
                        }
                    } else {
                        self.emit_save_acc_impl();
                        self.emit_load_ptr_from_slot_impl(slot, base.0);
                        if offset != 0 {
                            self.emit_add_offset_to_addr_reg_impl(offset);
                        }
                        self.emit_typed_store_indirect_impl(store_instr, ty);
                    }
                }
            }
        }
    }

    pub(super) fn emit_load_with_const_offset_impl(&mut self, dest: &Value, base: &Value, offset: i64, ty: IrType) {
        if ty == IrType::F128 {
            if let Some(addr) = self.state.resolve_slot_addr(base.0) {
                self.emit_f128_fldt(&addr, base.0, offset);
                self.emit_f128_load_finish(dest);
            }
            return;
        }
        let addr = self.state.resolve_slot_addr(base.0);
        if let Some(addr) = addr {
            let load_instr = Self::mov_load_for_type(ty);
            match addr {
                SlotAddr::OverAligned(slot, id) => {
                    self.emit_alloca_aligned_addr_impl(slot, id);
                    self.emit_add_offset_to_addr_reg_impl(offset);
                    self.emit_typed_load_indirect_impl(load_instr);
                }
                SlotAddr::Direct(slot) => {
                    let folded_slot = StackSlot(slot.0 + offset);
                    self.emit_typed_load_from_slot_impl(load_instr, folded_slot);
                }
                SlotAddr::Indirect(slot) => {
                    if let Some(&reg) = self.reg_assignments.get(&base.0) {
                        let reg_name = phys_reg_name(reg);
                        let dest_reg = Self::load_dest_reg(ty);
                        if offset != 0 {
                            self.state.emit_fmt(format_args!("    {} {}(%{}), {}", load_instr, offset, reg_name, dest_reg));
                        } else {
                            self.state.emit_fmt(format_args!("    {} (%{}), {}", load_instr, reg_name, dest_reg));
                        }
                    } else {
                        self.emit_load_ptr_from_slot_impl(slot, base.0);
                        if offset != 0 {
                            self.emit_add_offset_to_addr_reg_impl(offset);
                        }
                        self.emit_typed_load_indirect_impl(load_instr);
                    }
                }
            }
            self.store_rax_to(dest);
        }
    }

    pub(super) fn emit_typed_store_to_slot_impl(&mut self, instr: &'static str, ty: IrType, slot: StackSlot) {
        let reg = Self::reg_for_type("rax", ty);
        let out = &mut self.state.out;
        out.write_str("    ");
        out.write_str(instr);
        out.write_str(" %");
        out.write_str(reg);
        out.write_str(", ");
        if out.use_rsp_addressing {
            out.write_i64(out.rsp_frame_size + slot.0);
            out.write_str("(%rsp)");
        } else {
            out.write_i64(slot.0);
            out.write_str("(%rbp)");
        }
        out.newline();
    }

    pub(super) fn emit_typed_load_from_slot_impl(&mut self, instr: &'static str, slot: StackSlot) {
        let dest_reg = if instr == "movl" { "%eax" } else { "%rax" };
        let out = &mut self.state.out;
        out.write_str("    ");
        out.write_str(instr);
        out.write_str(" ");
        if out.use_rsp_addressing {
            out.write_i64(out.rsp_frame_size + slot.0);
            out.write_str("(%rsp), ");
        } else {
            out.write_i64(slot.0);
            out.write_str("(%rbp), ");
        }
        out.write_str(dest_reg);
        out.newline();
    }

    pub(super) fn emit_save_acc_impl(&mut self) {
        self.state.emit("    movq %rax, %rdx");
    }

    pub(super) fn emit_load_ptr_from_slot_impl(&mut self, slot: StackSlot, val_id: u32) {
        if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rcx");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rcx");
        }
    }

    pub(super) fn emit_typed_store_indirect_impl(&mut self, instr: &'static str, ty: IrType) {
        let store_reg = Self::reg_for_type("rdx", ty);
        self.state.emit_fmt(format_args!("    {} %{}, (%rcx)", instr, store_reg));
    }

    pub(super) fn emit_typed_load_indirect_impl(&mut self, instr: &'static str) {
        let dest_reg = if instr == "movl" { "%eax" } else { "%rax" };
        self.state.emit_fmt(format_args!("    {} (%rcx), {}", instr, dest_reg));
    }

    pub(super) fn emit_add_offset_to_addr_reg_impl(&mut self, offset: i64) {
        self.state.out.emit_instr_imm_reg("    addq", offset, "rcx");
    }

    /// Compute the address of an alloca into `reg`, handling over-aligned allocas.
    pub(super) fn emit_alloca_addr_to(&mut self, reg: &str, val_id: u32, offset: i64) {
        if let Some(align) = self.state.alloca_over_align(val_id) {
            self.state.out.emit_instr_rbp_reg("    leaq", offset, reg);
            self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, reg);
            self.state.out.emit_instr_imm_reg("    andq", -(align as i64), reg);
        } else {
            self.state.out.emit_instr_rbp_reg("    leaq", offset, reg);
        }
    }

    pub(super) fn emit_slot_addr_to_secondary_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr_to("rcx", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rcx");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rcx");
        }
    }

    pub(super) fn emit_add_secondary_to_acc_impl(&mut self) {
        self.state.emit("    addq %rcx, %rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_gep_direct_const_impl(&mut self, slot: StackSlot, offset: i64) {
        let folded = slot.0 + offset;
        self.state.out.emit_instr_rbp_reg("    leaq", folded, "rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_gep_indirect_const_impl(&mut self, slot: StackSlot, offset: i64, val_id: u32) {
        if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rax");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rax");
        }
        if offset != 0 {
            self.state.out.emit_instr_mem_reg("    leaq", offset, "rax", "rax");
        }
        self.state.reg_cache.invalidate_acc();
    }

    /// Emit leaq (%base, %index), %dest for GEP with both operands in registers.
    /// If dest is also register-allocated, emits directly to the dest register.
    /// Otherwise, emits to %rax and stores via store_rax_to.
    pub(super) fn emit_leaq_base_index_impl(
        &mut self,
        base_reg: super::super::super::regalloc::PhysReg,
        index_reg: super::super::super::regalloc::PhysReg,
        dest: &Value,
        dest_phys: Option<super::super::super::regalloc::PhysReg>,
    ) {
        use super::emit::{phys_reg_name};
        let base_name = phys_reg_name(base_reg);
        let index_name = phys_reg_name(index_reg);

        if let Some(dp) = dest_phys {
            let dest_name = phys_reg_name(dp);
            self.state.emit_fmt(format_args!("    leaq (%{}, %{}), %{}", base_name, index_name, dest_name));
        } else {
            self.state.emit_fmt(format_args!("    leaq (%{}, %{}), %rax", base_name, index_name));
            self.store_rax_to(dest);
        }
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_gep_add_const_to_acc_impl(&mut self, offset: i64) {
        if offset != 0 {
            self.state.out.emit_instr_imm_reg("    addq", offset, "rax");
        }
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_add_imm_to_acc_impl(&mut self, imm: i64) {
        self.state.out.emit_instr_imm_reg("    addq", imm, "rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_round_up_acc_to_16_impl(&mut self) {
        self.state.emit("    addq $15, %rax");
        self.state.emit("    andq $-16, %rax");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_sub_sp_by_acc_impl(&mut self) {
        self.state.emit("    subq %rax, %rsp");
    }

    pub(super) fn emit_mov_sp_to_acc_impl(&mut self) {
        self.state.emit("    movq %rsp, %rax");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_mov_acc_to_sp_impl(&mut self) {
        self.state.emit("    movq %rax, %rsp");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_align_acc_impl(&mut self, align: usize) {
        self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rax");
        self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rax");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_memcpy_load_dest_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr_to("rdi", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rdi");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rdi");
        }
    }

    pub(super) fn emit_memcpy_load_src_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) {
        if is_alloca {
            self.emit_alloca_addr_to("rsi", val_id, slot.0);
        } else if let Some(&reg) = self.reg_assignments.get(&val_id) {
            let reg_name = phys_reg_name(reg);
            self.state.out.emit_instr_reg_reg("    movq", reg_name, "rsi");
        } else {
            self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rsi");
        }
    }

    pub(super) fn emit_alloca_aligned_addr_impl(&mut self, slot: StackSlot, val_id: u32) {
        let align = self.state.alloca_over_align(val_id)
            .expect("alloca must have over-alignment for aligned addr emission");
        self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rcx");
        self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rcx");
        self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rcx");
    }

    pub(super) fn emit_alloca_aligned_addr_to_acc_impl(&mut self, slot: StackSlot, val_id: u32) {
        let align = self.state.alloca_over_align(val_id)
            .expect("alloca must have over-alignment for aligned addr emission");
        self.state.out.emit_instr_rbp_reg("    leaq", slot.0, "rax");
        self.state.out.emit_instr_imm_reg("    addq", (align - 1) as i64, "rax");
        self.state.out.emit_instr_imm_reg("    andq", -(align as i64), "rax");
        self.state.reg_cache.invalidate_acc();
    }

    pub(super) fn emit_acc_to_secondary_impl(&mut self) {
        self.state.emit("    movq %rax, %rcx");
    }

    pub(super) fn emit_memcpy_store_dest_from_acc_impl(&mut self) {
        self.state.emit("    movq %rcx, %rdi");
    }

    pub(super) fn emit_memcpy_store_src_from_acc_impl(&mut self) {
        self.state.emit("    movq %rcx, %rsi");
    }

    pub(super) fn emit_memcpy_impl_impl(&mut self, size: usize) {
        self.state.out.emit_instr_imm_reg("    movq", size as i64, "rcx");
        self.state.emit("    rep movsb");
    }

    // ---- Segment-prefixed memory ops ----

    pub(super) fn emit_seg_load_impl(&mut self, dest: &Value, ptr: &Value, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        self.operand_to_rax(&Operand::Value(*ptr));
        self.state.emit("    movq %rax, %rcx");
        let load_instr = Self::mov_load_for_type(ty);
        let dest_reg = Self::load_dest_reg(ty);
        self.state.emit_fmt(format_args!("    {} {}(%rcx), {}", load_instr, seg_prefix, dest_reg));
        self.store_rax_to(dest);
    }

    pub(super) fn emit_seg_load_symbol_impl(&mut self, dest: &Value, sym: &str, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        let load_instr = Self::mov_load_for_type(ty);
        let dest_reg = Self::load_dest_reg(ty);
        self.state.emit_fmt(format_args!("    {} {}{}(%rip), {}", load_instr, seg_prefix, sym, dest_reg));
        self.store_rax_to(dest);
    }

    pub(super) fn emit_seg_store_impl(&mut self, val: &Operand, ptr: &Value, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        self.operand_to_rax(val);
        self.state.emit("    movq %rax, %rdx");
        self.operand_to_rax(&Operand::Value(*ptr));
        self.state.emit("    movq %rax, %rcx");
        let store_instr = Self::mov_store_for_type(ty);
        let store_reg = Self::reg_for_type("rdx", ty);
        self.state.emit_fmt(format_args!("    {} %{}, {}(%rcx)", store_instr, store_reg, seg_prefix));
    }

    pub(super) fn emit_seg_store_symbol_impl(&mut self, val: &Operand, sym: &str, ty: IrType, seg: AddressSpace) {
        let seg_prefix = match seg {
            AddressSpace::SegGs => "%gs:",
            AddressSpace::SegFs => "%fs:",
            AddressSpace::Default => unreachable!("segment-prefixed op called with default address space"),
        };
        self.operand_to_rax(val);
        let store_instr = Self::mov_store_for_type(ty);
        let store_reg = Self::reg_for_type("rax", ty);
        self.state.emit_fmt(format_args!("    {} %{}, {}{}(%rip)", store_instr, store_reg, seg_prefix, sym));
    }
}
