//! X86Codegen: cast operations.

use crate::ir::reexports::{IrConst, Operand, Value};
use crate::common::types::IrType;
use crate::backend::generation::is_i128_type;
use super::emit::X86Codegen;

impl X86Codegen {
    pub(super) fn emit_cast_instrs_impl(&mut self, from_ty: IrType, to_ty: IrType) {
        self.emit_cast_instrs_x86(from_ty, to_ty);
    }

    pub(super) fn emit_cast_impl(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) {
        // Intercept casts TO F128: produce full 80-bit x87 value in dest slot.
        if to_ty == IrType::F128 && from_ty != IrType::F128 && !is_i128_type(from_ty) {
            if let Some(dest_slot) = self.state.get_slot(dest.0) {
                if from_ty == IrType::F64 {
                    self.operand_to_rax(src);
                    self.state.emit("    subq $8, %rsp");
                    self.state.emit("    movq %rax, (%rsp)");
                    self.state.emit("    fldl (%rsp)");
                    self.state.emit("    addq $8, %rsp");
                } else if from_ty == IrType::F32 {
                    self.operand_to_rax(src);
                    self.state.emit("    subq $4, %rsp");
                    self.state.emit("    movl %eax, (%rsp)");
                    self.state.emit("    flds (%rsp)");
                    self.state.emit("    addq $4, %rsp");
                } else if from_ty.is_signed() || (!from_ty.is_float() && !from_ty.is_unsigned()) {
                    self.operand_to_rax(src);
                    if from_ty.size() < 8 {
                        self.emit_cast_instrs_x86(from_ty, IrType::I64);
                    }
                    self.state.emit("    subq $8, %rsp");
                    self.state.emit("    movq %rax, (%rsp)");
                    self.state.emit("    fildq (%rsp)");
                    self.state.emit("    addq $8, %rsp");
                } else {
                    self.operand_to_rax(src);
                    if from_ty.size() < 8 {
                        self.emit_cast_instrs_x86(from_ty, IrType::I64);
                    }
                    let big_label = self.state.fresh_label("u2f128_big");
                    let done_label = self.state.fresh_label("u2f128_done");
                    self.state.emit("    testq %rax, %rax");
                    self.state.out.emit_jcc_label("    js", &big_label);
                    self.state.emit("    subq $8, %rsp");
                    self.state.emit("    movq %rax, (%rsp)");
                    self.state.emit("    fildq (%rsp)");
                    self.state.emit("    addq $8, %rsp");
                    self.state.out.emit_jmp_label(&done_label);
                    self.state.out.emit_named_label(&big_label);
                    self.state.emit("    subq $8, %rsp");
                    self.state.emit("    movq %rax, (%rsp)");
                    self.state.emit("    fildq (%rsp)");
                    self.state.emit("    addq $8, %rsp");
                    self.state.emit("    subq $16, %rsp");
                    self.state.out.emit_instr_imm_reg("    movabsq", -9223372036854775808i64, "rax");
                    self.state.emit("    movq %rax, (%rsp)");
                    self.state.out.emit_instr_imm_reg("    movq", 0x403Fi64, "rax");
                    self.state.emit("    movq %rax, 8(%rsp)");
                    self.state.emit("    fldt (%rsp)");
                    self.state.emit("    addq $16, %rsp");
                    self.state.emit("    faddp %st, %st(1)");
                    self.state.out.emit_named_label(&done_label);
                }
                self.state.out.emit_instr_rbp("    fstpt", dest_slot.0);
                self.state.out.emit_instr_rbp("    fldt", dest_slot.0);
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.state.reg_cache.set_acc(dest.0, false);
                self.state.f128_direct_slots.insert(dest.0);
                return;
            }
        }

        // Intercept F128 -> F64/F32 casts
        if from_ty == IrType::F128 && (to_ty == IrType::F64 || to_ty == IrType::F32) {
            self.emit_f128_load_to_x87(src);
            if to_ty == IrType::F64 {
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    movq (%rsp), %rax");
                self.state.emit("    addq $8, %rsp");
            } else {
                self.state.emit("    subq $4, %rsp");
                self.state.emit("    fstps (%rsp)");
                self.state.emit("    movl (%rsp), %eax");
                self.state.emit("    addq $4, %rsp");
            }
            self.state.reg_cache.invalidate_acc();
            self.store_rax_to(dest);
            return;
        }

        // Intercept F128 -> integer casts when we know the source's memory location
        if from_ty == IrType::F128 && !to_ty.is_float() && !is_i128_type(to_ty) {
            if let Operand::Value(v) = src {
                if self.state.f128_direct_slots.contains(&v.0) {
                    if let Some(slot) = self.state.get_slot(v.0) {
                        let addr = crate::backend::state::SlotAddr::Direct(slot);
                        self.emit_f128_to_int_from_memory(&addr, to_ty);
                        self.store_rax_to(dest);
                        return;
                    }
                }
                if let Some((ptr_id, _offset, _is_indirect)) = self.state.get_f128_source(v.0) {
                    if let Some(addr) = self.state.resolve_slot_addr(ptr_id) {
                        self.emit_f128_to_int_from_memory(&addr, to_ty);
                        self.store_rax_to(dest);
                        return;
                    }
                }
            }
            if let Operand::Const(IrConst::LongDouble(_, f128_bytes)) = src {
                let x87 = crate::common::long_double::f128_bytes_to_x87_bytes(f128_bytes);
                self.state.emit("    subq $16, %rsp");
                let lo = u64::from_le_bytes(x87[0..8].try_into().unwrap());
                let hi = u16::from_le_bytes(x87[8..10].try_into().unwrap());
                self.state.out.emit_instr_imm_reg("    movabsq", lo as i64, "rax");
                self.state.emit("    movq %rax, (%rsp)");
                self.state.out.emit_instr_imm_reg("    movq", hi as i64, "rax");
                self.state.emit("    movq %rax, 8(%rsp)");
                self.state.emit("    fldt (%rsp)");
                self.state.emit("    addq $16, %rsp");
                self.emit_f128_st0_to_int(to_ty);
                self.store_rax_to(dest);
                return;
            }
        }
        // Register-direct integer casts: bypass the accumulator when the destination
        // has a physical register. Instead of load→cast→store through %rax, emit the
        // cast instruction directly targeting the destination register.
        if !from_ty.is_float() && !to_ty.is_float()
            && !is_i128_type(from_ty) && !is_i128_type(to_ty)
        {
            if let Some(dest_phys) = self.dest_reg(dest) {
                if !super::emit::is_xmm_reg(dest_phys) {
                    if self.try_emit_cast_reg_direct(dest, src, from_ty, to_ty, dest_phys) {
                        return;
                    }
                }
            }
        }

        // Fall through to default implementation for all other cases
        crate::backend::traits::emit_cast_default(self, dest, src, from_ty, to_ty);
    }

    /// Try to emit an integer cast directly to the destination register.
    /// Returns true if the cast was emitted, false if we should fall through.
    ///
    /// Key optimization: when the source is a stack slot, fuse the load and cast
    /// into a single instruction (e.g., `movslq -N(%rsp), %r12`), saving 2 instructions
    /// vs the accumulator path (`movq -N(%rsp), %rax; cltq; movq %rax, %r12`).
    fn try_emit_cast_reg_direct(
        &mut self,
        _dest: &Value,
        src: &Operand,
        from_ty: IrType,
        to_ty: IrType,
        dest_phys: crate::backend::regalloc::PhysReg,
    ) -> bool {
        use crate::backend::cast::{classify_cast, CastKind};
        use super::emit::{phys_reg_name, phys_reg_name_32, typed_phys_reg_name};

        let dest_64 = phys_reg_name(dest_phys);
        let dest_32 = phys_reg_name_32(dest_phys);
        let kind = classify_cast(from_ty, to_ty);

        // Resolve source: either a physical register or a stack slot offset.
        let src_phys = self.operand_reg(src);
        let src_slot = match src {
            Operand::Value(v) => self.state.get_slot(v.0).map(|s| s.0),
            _ => None,
        };

        // For constants, fall through to default path (it's already efficient).
        if matches!(src, Operand::Const(_)) {
            return false;
        }

        // Need either a source register or a stack slot.
        if src_phys.is_none() && src_slot.is_none() {
            return false;
        }

        match kind {
            CastKind::Noop | CastKind::UnsignedToSignedSameSize { .. } => {
                // No conversion needed — just move src to dest register.
                self.operand_to_callee_reg(src, dest_phys);
                return true;
            }
            CastKind::IntWiden { from_ty: ft, .. } => {
                // Widen: sign or zero extend from smaller to larger type.
                if let Some(src_reg) = src_phys {
                    // Source is in a register — emit reg-to-reg extending move.
                    let src_typed = typed_phys_reg_name(src_reg, ft);
                    if ft.is_signed() {
                        match ft.size() {
                            1 => self.state.emit_fmt(format_args!("    movsbq %{}, %{}", src_typed, dest_64)),
                            2 => self.state.emit_fmt(format_args!("    movswq %{}, %{}", src_typed, dest_64)),
                            4 => self.state.emit_fmt(format_args!("    movslq %{}, %{}", phys_reg_name_32(src_reg), dest_64)),
                            _ => { self.operand_to_callee_reg(src, dest_phys); }
                        }
                    } else {
                        match ft.size() {
                            1 => self.state.emit_fmt(format_args!("    movzbl %{}, %{}", src_typed, dest_32)),
                            2 => self.state.emit_fmt(format_args!("    movzwl %{}, %{}", src_typed, dest_32)),
                            4 => {
                                let src_32 = phys_reg_name_32(src_reg);
                                self.state.emit_fmt(format_args!("    movl %{}, %{}", src_32, dest_32));
                            }
                            _ => { self.operand_to_callee_reg(src, dest_phys); }
                        }
                    }
                } else if let Some(slot_off) = src_slot {
                    // Source is a stack slot — emit fused load+extend directly to dest.
                    // Uses emit_instr_rbp_reg which handles rbp/rsp addressing automatically.
                    if ft.is_signed() {
                        match ft.size() {
                            1 => self.state.out.emit_instr_rbp_reg("    movsbq", slot_off, dest_64),
                            2 => self.state.out.emit_instr_rbp_reg("    movswq", slot_off, dest_64),
                            4 => self.state.out.emit_instr_rbp_reg("    movslq", slot_off, dest_64),
                            _ => self.state.out.emit_instr_rbp_reg("    movq", slot_off, dest_64),
                        }
                    } else {
                        match ft.size() {
                            1 => self.state.out.emit_instr_rbp_reg("    movzbl", slot_off, dest_32),
                            2 => self.state.out.emit_instr_rbp_reg("    movzwl", slot_off, dest_32),
                            4 => self.state.out.emit_instr_rbp_reg("    movl", slot_off, dest_32),
                            _ => self.state.out.emit_instr_rbp_reg("    movq", slot_off, dest_64),
                        }
                    }
                } else {
                    return false;
                }
                return true;
            }
            CastKind::IntNarrow { to_ty: t } => {
                // Narrow: truncate. Load full value then mask/zero-extend.
                if let Some(src_reg) = src_phys {
                    let src_typed = typed_phys_reg_name(src_reg, t);
                    match t.size() {
                        4 => self.state.emit_fmt(format_args!("    movl %{}, %{}", phys_reg_name_32(src_reg), dest_32)),
                        2 => self.state.emit_fmt(format_args!("    movzwl %{}, %{}", src_typed, dest_32)),
                        1 => self.state.emit_fmt(format_args!("    movzbl %{}, %{}", src_typed, dest_32)),
                        _ => { self.operand_to_callee_reg(src, dest_phys); }
                    }
                } else if let Some(slot_off) = src_slot {
                    // For narrowing from stack: just load the narrower size.
                    match t.size() {
                        4 => self.state.out.emit_instr_rbp_reg("    movl", slot_off, dest_32),
                        2 => self.state.out.emit_instr_rbp_reg("    movzwl", slot_off, dest_32),
                        1 => self.state.out.emit_instr_rbp_reg("    movzbl", slot_off, dest_32),
                        _ => self.state.out.emit_instr_rbp_reg("    movq", slot_off, dest_64),
                    }
                } else {
                    return false;
                }
                return true;
            }
            CastKind::SignedToUnsignedSameSize { to_ty: t } => {
                // Same size, just zero-extend to clear sign bits.
                if let Some(src_reg) = src_phys {
                    match t.size() {
                        4 => self.state.emit_fmt(format_args!("    movl %{}, %{}", phys_reg_name_32(src_reg), dest_32)),
                        2 => self.state.emit_fmt(format_args!("    movzwl %{}, %{}", typed_phys_reg_name(src_reg, t), dest_32)),
                        1 => self.state.emit_fmt(format_args!("    movzbl %{}, %{}", typed_phys_reg_name(src_reg, t), dest_32)),
                        _ => { self.operand_to_callee_reg(src, dest_phys); }
                    }
                } else if let Some(slot_off) = src_slot {
                    match t.size() {
                        4 => self.state.out.emit_instr_rbp_reg("    movl", slot_off, dest_32),
                        2 => self.state.out.emit_instr_rbp_reg("    movzwl", slot_off, dest_32),
                        1 => self.state.out.emit_instr_rbp_reg("    movzbl", slot_off, dest_32),
                        _ => self.state.out.emit_instr_rbp_reg("    movq", slot_off, dest_64),
                    }
                } else {
                    return false;
                }
                return true;
            }
            _ => {
                // Float casts, F128, etc. — not handled here.
                return false;
            }
        }
    }
}
