//! X86Codegen: prologue, epilogue, parameter storage.

use crate::ir::reexports::{IrBinOp, IrFunction, Instruction, Operand, Terminator, Value};
use crate::common::types::IrType;
use crate::common::fx_hash::FxHashSet;
use crate::backend::call_abi::{ParamClass, classify_params};
use crate::backend::generation::{calculate_stack_space_common, find_param_alloca};
use crate::backend::regalloc::PhysReg;
use super::emit::{X86Codegen, X86_CALLEE_SAVED, X86_CALLEE_SAVED_WITH_RBP, X86_CALLER_SAVED,
                     phys_reg_name, collect_inline_asm_callee_saved_x86, X86_ARG_REGS};

impl X86Codegen {
    pub(super) fn calculate_stack_space_impl(&mut self, func: &IrFunction) -> i64 {
        // Store function pointer for indexed addressing detection
        self.current_func = Some(func as *const IrFunction);

        // Analyze IVSR patterns for Phase 9b indexed addressing optimization
        self.analyze_ivsr_pointers(func);

        // Track variadic function info
        self.is_variadic = func.is_variadic;
        // Count named params using the shared ABI classification, so this
        // stays in sync with classify_call_args (caller side) automatically.
        {
            let config = self.call_abi_config_impl();
            let classification = crate::backend::call_abi::classify_params_full(func, &config);
            let mut named_gp = 0usize;
            let mut named_fp = 0usize;
            for class in &classification.classes {
                named_gp += class.gp_reg_count();
                if matches!(class, crate::backend::call_abi::ParamClass::FloatReg { .. }) {
                    named_fp += 1;
                }
            }
            self.num_named_int_params = named_gp;
            self.num_named_fp_params = named_fp;
            self.num_named_stack_bytes =
                crate::backend::call_abi::named_params_stack_bytes(&classification.classes);
        }

        // Run register allocator BEFORE stack space computation so we can
        // skip allocating stack slots for values assigned to registers.
        let mut asm_clobbered_regs: Vec<PhysReg> = Vec::new();
        collect_inline_asm_callee_saved_x86(func, &mut asm_clobbered_regs);
        let callee_base: &[PhysReg] = if self.state.omit_frame_pointer {
            &X86_CALLEE_SAVED_WITH_RBP
        } else {
            &X86_CALLEE_SAVED
        };
        let available_regs = crate::backend::generation::filter_available_regs(callee_base, &asm_clobbered_regs);

        let mut caller_saved_regs = X86_CALLER_SAVED.to_vec();
        let mut has_indirect_call = false;
        let mut has_i128_ops = false;
        let mut has_atomic_rmw = false;
        // Track rdx-clobbering patterns for conditional rdx allocation
        let mut has_div_rem = false;
        let mut has_gep = false;       // GEP → indirect stores → emit_save_acc uses rdx
        let mut has_switch = false;    // Switch → jump tables use rdx
        let mut has_select = false;    // Select → cmov path uses rdx
        let mut has_i32_widening = false; // Cast from I32/U32 to I64/pointer → needs sign-ext
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::CallIndirect { .. } => { has_indirect_call = true; }
                    Instruction::BinOp { op, ty, .. } => {
                        if matches!(ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                        if matches!(op, IrBinOp::SDiv | IrBinOp::UDiv | IrBinOp::SRem | IrBinOp::URem) {
                            has_div_rem = true;
                        }
                    }
                    Instruction::UnaryOp { ty, .. } => {
                        if matches!(ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                    }
                    Instruction::Cast { from_ty, to_ty, .. } => {
                        if matches!(from_ty, IrType::I128 | IrType::U128)
                            || matches!(to_ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                        // Detect I32/U32 widening to 64-bit: requires sign-extension.
                        if matches!(from_ty, IrType::I32 | IrType::U32)
                            && matches!(to_ty, IrType::I64 | IrType::U64 | IrType::Ptr) {
                            has_i32_widening = true;
                        }
                    }
                    Instruction::Cmp { ty, .. }
                    | Instruction::Store { ty, .. } => {
                        if matches!(ty, IrType::I128 | IrType::U128) {
                            has_i128_ops = true;
                        }
                    }
                    Instruction::AtomicRmw { .. } => { has_atomic_rmw = true; }
                    Instruction::GetElementPtr { .. } => { has_gep = true; }
                    Instruction::Select { .. } => { has_select = true; }
                    _ => {}
                }
            }
            if matches!(block.terminator, Terminator::Switch { .. }) {
                has_switch = true;
            }
        }
        if has_indirect_call {
            caller_saved_regs.retain(|r| r.0 != 11); // r10 = PhysReg(11)
        }
        if has_i128_ops {
            caller_saved_regs.retain(|r| r.0 != 12 && r.0 != 13 && r.0 != 14 && r.0 != 15); // r8, r9, rdi, rsi
        }
        if has_atomic_rmw {
            caller_saved_regs.retain(|r| r.0 != 12); // r8
        }
        // rdx (PhysReg 16) is available as caller-saved when no codegen path uses
        // it as scratch: no div/rem (implicit rdx), no i128 (rax:rdx pair), no GEP
        // (indirect stores save acc to rdx), no switch (jump tables use rdx), no
        // select (cmov path zeros rdx).
        if !has_div_rem && !has_i128_ops && !has_gep && !has_switch && !has_select {
            caller_saved_regs.push(PhysReg(16)); // rdx
        }

        // Build set of I32 values that need sign-extension (used in 64-bit contexts).
        // Values NOT in this set can skip movslq after 32-bit register ALU ops.
        // A value needs sext if it's: (a) a Cast source going to I64/U64/Ptr,
        // (b) an operand of GetElementPtr, or (c) used in a 64-bit BinOp/Cmp.
        let mut needs_sext_set: FxHashSet<u32> = FxHashSet::default();
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::Cast { src, from_ty, to_ty, .. } => {
                        if matches!(from_ty, IrType::I32 | IrType::U32)
                            && matches!(to_ty, IrType::I64 | IrType::U64 | IrType::Ptr) {
                            if let Operand::Value(v) = src {
                                needs_sext_set.insert(v.0);
                            }
                        }
                    }
                    Instruction::GetElementPtr { base, offset, .. } => {
                        needs_sext_set.insert(base.0);
                        if let Operand::Value(v) = offset {
                            needs_sext_set.insert(v.0);
                        }
                    }
                    _ => {}
                }
            }
        }
        self.skip_i32_sext = !has_gep && needs_sext_set.is_empty();
        self.needs_sext_values = needs_sext_set;

        let (reg_assigned, cached_liveness) = crate::backend::generation::run_regalloc_and_merge_clobbers(
            func, available_regs, caller_saved_regs, &asm_clobbered_regs,
            &mut self.reg_assignments, &mut self.used_callee_saved,
            false,
        );

        // FPO (RSP mode): callee saves are movq'd into the frame at offsets -8..-N*8
        // from the virtual rbp. callee_save_reserve shifts local slots below them.
        // RBP (push mode): pushes go BEFORE subq and are at -8(%rbp)..-N*8(%rbp).
        // Local slots are within the subq frame starting at -(N*8+first_slot)(%rbp).
        // For non-variadic RBP functions, no reserve is needed because the subq frame
        // starts below the push area. For variadic functions, the register save area
        // is added to `space` separately.
        let callee_save_reserve = (self.used_callee_saved.len() as i64) * 8;
        let mut space = calculate_stack_space_common(&mut self.state, func, callee_save_reserve, |space, alloc_size, align| {
            let effective_align = if align > 0 { align.max(8) } else { 8 };
            let alloc = (alloc_size + 7) & !7;
            let new_space = ((space + alloc + effective_align - 1) / effective_align) * effective_align;
            (-new_space, new_space)
        }, &reg_assigned, &X86_CALLEE_SAVED, cached_liveness, false);

        if func.is_variadic {
            if self.no_sse {
                space += 48;
            } else {
                space += 176;
            }
            self.reg_save_area_offset = -space;
        }

        // `space` includes callee_save_reserve for the save area — return as-is.
        space
    }

    pub(super) fn aligned_frame_size_impl(&self, raw_space: i64) -> i64 {
        if raw_space <= 0 { return 0; }
        let aligned = (raw_space + 15) & !15;
        if self.state.omit_frame_pointer {
            // With frame pointer omission, there's no `push %rbp` to absorb the
            // 8-byte return address misalignment. The frame size must be ≡ 8 (mod 16)
            // so that RSP is 16-byte aligned at subsequent CALL instructions.
            // At function entry: RSP ≡ 8 (mod 16) due to the caller's CALL.
            // subq $(8 mod 16), %rsp → RSP ≡ 8 - 8 = 0 (mod 16) ✓
            if aligned % 16 == 0 { aligned + 8 } else { aligned }
        } else {
            aligned
        }
    }

    pub(super) fn emit_prologue_impl(&mut self, func: &IrFunction, frame_size: i64) {
        self.current_return_type = func.return_type;
        self.func_ret_classes = func.ret_eightbyte_classes.clone();
        if self.state.cf_protection_branch {
            self.state.emit("    endbr64");
        }

        let omit_fp = self.state.omit_frame_pointer;
        let used_regs = self.used_callee_saved.clone();

        if omit_fp {
            // Frame-pointer-less prologue: allocate entire frame with subq,
            // then save callee-saved registers into the frame using movq.
            // Stack layout (RSP-relative):
            //   [return addr]  ← old RSP
            //   [callee saves] ← top of frame
            //   [local vars]   ← RSP points here after subq
            if frame_size > 0 {
                self.state.out.emit_instr_imm_reg("    subq", frame_size, "rsp");
                if self.state.emit_cfi {
                    self.state.emit_fmt(format_args!("    .cfi_def_cfa_offset {}", frame_size + 8));
                }
            }
            // Save callee-saved registers at the same offsets the push prologue
            // would use: -8(%rbp), -16(%rbp), etc. These translate to
            // (frame_size-8)(%rsp), (frame_size-16)(%rsp), etc.
            // These offsets overlap with the stack layout's register-allocated
            // value slots, which is correct — the stack layout assigned those
            // slots to the callee-saved-register values, so writing the register
            // value there is the intended initialization.
            for (i, &reg) in used_regs.iter().enumerate() {
                let reg_name = phys_reg_name(reg);
                let rbp_offset = -((i as i64 + 1) * 8);
                let rsp_offset = frame_size + rbp_offset;
                self.state.emit_fmt(format_args!("    movq %{}, {}(%rsp)", reg_name, rsp_offset));
            }
            // Set AsmOutput to use RSP-relative addressing
            self.state.out.use_rsp_addressing = true;
            self.state.out.rsp_frame_size = frame_size;
        } else {
            // Traditional frame-pointer prologue.
            // Reset RSP-relative addressing flag: a previous FPO function may have
            // set it, and the AsmOutput is shared across all functions in the module.
            self.state.out.use_rsp_addressing = false;
            self.state.emit("    pushq %rbp");
            if self.state.emit_cfi {
                self.state.emit("    .cfi_def_cfa_offset 16");
                self.state.emit("    .cfi_offset %rbp, -16");
            }
            self.state.emit("    movq %rsp, %rbp");
            if self.state.emit_cfi {
                self.state.emit("    .cfi_def_cfa_register %rbp");
            }

            // Save callee-saved registers with pushq
            for &reg in &used_regs {
                let reg_name = phys_reg_name(reg);
                self.state.emit_fmt(format_args!("    pushq %{}", reg_name));
            }

            // Allocate remaining stack space for local variables
            let local_size = frame_size - (used_regs.len() as i64 * 8);
            if local_size > 0 {
                const PAGE_SIZE: i64 = 4096;
                if local_size > PAGE_SIZE {
                    let probe_label = self.state.fresh_label("stack_probe");
                    self.state.out.emit_instr_imm_reg("    movq", local_size, "r11");
                    self.state.out.emit_named_label(&probe_label);
                    self.state.out.emit_instr_imm_reg("    subq", PAGE_SIZE, "rsp");
                    self.state.emit("    orl $0, (%rsp)");
                    self.state.out.emit_instr_imm_reg("    subq", PAGE_SIZE, "r11");
                    self.state.out.emit_instr_imm_reg("    cmpq", PAGE_SIZE, "r11");
                    self.state.out.emit_jcc_label("    ja", &probe_label);
                    self.state.emit("    subq %r11, %rsp");
                    self.state.emit("    orl $0, (%rsp)");
                } else {
                    self.state.out.emit_instr_imm_reg("    subq", local_size, "rsp");
                }
            }
        }

        if func.is_variadic {
            let base = self.reg_save_area_offset;
            self.state.out.emit_instr_reg_rbp("    movq", "rdi", base);
            self.state.out.emit_instr_reg_rbp("    movq", "rsi", base + 8);
            self.state.out.emit_instr_reg_rbp("    movq", "rdx", base + 16);
            self.state.out.emit_instr_reg_rbp("    movq", "rcx", base + 24);
            self.state.out.emit_instr_reg_rbp("    movq", "r8", base + 32);
            self.state.out.emit_instr_reg_rbp("    movq", "r9", base + 40);
            if !self.no_sse {
                for i in 0..8i64 {
                    self.state.emit_fmt(format_args!("    movdqu %xmm{}, {}(%rbp)", i, base + 48 + i * 16));
                }
            }
        }
    }

    pub(super) fn emit_epilogue_impl(&mut self, frame_size: i64) {
        let used_regs = self.used_callee_saved.clone();
        let num_saved = used_regs.len() as i64;
        let omit_fp = self.state.omit_frame_pointer;

        if omit_fp {
            // Frame-pointer-less epilogue: restore callee-saved regs from stack, then addq
            for (i, &reg) in used_regs.iter().enumerate() {
                let reg_name = phys_reg_name(reg);
                let offset = frame_size - (i as i64 + 1) * 8;
                self.state.emit_fmt(format_args!("    movq {}(%rsp), %{}", offset, reg_name));
            }
            if frame_size > 0 {
                self.state.out.emit_instr_imm_reg("    addq", frame_size, "rsp");
            }
        } else {
            // Traditional epilogue: restore from pushes, then popq %rbp
            if num_saved > 0 {
                self.state.emit_fmt(format_args!("    leaq {}(%rbp), %rsp", -(num_saved * 8)));
            } else {
                self.state.emit("    movq %rbp, %rsp");
            }
            for &reg in used_regs.iter().rev() {
                let reg_name = phys_reg_name(reg);
                self.state.emit_fmt(format_args!("    popq %{}", reg_name));
            }
            self.state.emit("    popq %rbp");
        }
        let _ = frame_size;
    }

    pub(super) fn emit_store_params_impl(&mut self, func: &IrFunction) {
        let xmm_regs = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"];
        let config = self.call_abi_config_impl();
        let param_classes = classify_params(func, &config);
        self.state.param_classes = param_classes.clone();
        self.state.num_params = func.params.len();
        self.state.func_is_variadic = func.is_variadic;

        self.state.param_alloca_slots = (0..func.params.len()).map(|i| {
            find_param_alloca(func, i).and_then(|(dest, ty)| {
                self.state.get_slot(dest.0).map(|slot| (slot, ty))
            })
        }).collect();

        // Build a map of param_idx -> ParamRef dest Value for fast lookup.
        // This is used to optimize parameter storage: when the ParamRef dest
        // is register-allocated, we can store the ABI arg register directly
        // to the callee-saved register, skipping the alloca slot entirely.
        let mut paramref_dests: Vec<Option<Value>> = vec![None; func.params.len()];
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::ParamRef { dest, param_idx, .. } = inst {
                    if *param_idx < paramref_dests.len() {
                        paramref_dests[*param_idx] = Some(*dest);
                    }
                }
            }
        }

        // In RBP mode: stack args start at 16(%rbp) = return_addr(8) + saved_rbp(8).
        // In FPO mode: no saved rbp, so stack args start at 8 from the virtual rbp
        // (which is entry_rsp = return_addr).
        let stack_base: i64 = if self.state.omit_frame_pointer { 8 } else { 16 };

        // Build a map from physical register -> list of param indices that use it,
        // so we can detect when two params share the same callee-saved register.
        let mut reg_to_params: crate::common::fx_hash::FxHashMap<u8, Vec<usize>> = crate::common::fx_hash::FxHashMap::default();
        for (i, _) in func.params.iter().enumerate() {
            if let Some(paramref_dest) = paramref_dests[i] {
                if let Some(&phys_reg) = self.reg_assignments.get(&paramref_dest.0) {
                    reg_to_params.entry(phys_reg.0).or_default().push(i);
                }
            }
        }

        for (i, _param) in func.params.iter().enumerate() {
            let class = param_classes[i];

            // Pre-store optimization: when a param's alloca is dead (eliminated by
            // dead param alloca analysis) but the ParamRef dest is register-assigned,
            // store the ABI arg register directly to the assigned physical register
            // in the prologue. This is critical because:
            // 1. Dead alloca means no stack slot exists for this param
            // 2. The ABI register (rdi, rsi, etc.) is caller-saved and will be clobbered
            // 3. We must save the value NOW, before any other code runs
            // 4. emit_param_ref will see param_pre_stored and skip code generation
            if let Some(paramref_dest) = paramref_dests[i] {
                let has_slot = find_param_alloca(func, i)
                    .and_then(|(dest, _)| self.state.get_slot(dest.0))
                    .is_some();
                if !has_slot {
                    if let Some(&phys_reg) = self.reg_assignments.get(&paramref_dest.0) {
                        // Only pre-store to callee-saved registers (PhysReg 1-5).
                        // Caller-saved registers (rdi, rsi, r8-r11) cannot be used
                        // because they overlap with ABI argument registers that
                        // haven't been saved yet.
                        let is_callee_saved = phys_reg.0 >= 1 && phys_reg.0 <= 5;
                        if is_callee_saved {
                            // Safety check: if another param's dest is also assigned
                            // to this register, skip pre-store to avoid conflicts.
                            let shared = reg_to_params.get(&phys_reg.0)
                                .is_some_and(|users| users.len() > 1);
                            if !shared {
                                let dest_reg = phys_reg_name(phys_reg);
                                if let ParamClass::IntReg { reg_idx } = class {
                                    self.state.out.emit_instr_reg_reg(
                                        "    movq", X86_ARG_REGS[reg_idx], dest_reg);
                                    self.state.param_pre_stored.insert(i);
                                } // TODO: handle StackSlot/SSE params
                            }
                        }
                    }
                    continue;
                }
            }

            let (slot, ty) = if let Some((dest, ty)) = find_param_alloca(func, i) {
                if let Some(slot) = self.state.get_slot(dest.0) {
                    (slot, ty)
                } else {
                    continue;
                }
            } else {
                continue;
            };

            match class {
                ParamClass::IntReg { reg_idx } => {
                    // Always store the full 64-bit register to ensure the entire 8-byte
                    // slot is initialized. Using a typed store (e.g., movl for I32) would
                    // only write 4 bytes, leaving the upper bytes uninitialized. Later
                    // untyped loads via value_to_reg use movq (8 bytes), which would read
                    // uninitialized memory and trigger valgrind errors.
                    // The typed load in emit_param_ref_impl correctly extracts only the
                    // meaningful bytes (e.g., movslq for I32).
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[reg_idx], slot.0);
                }
                ParamClass::FloatReg { reg_idx } => {
                    if ty == IrType::F32 {
                        self.state.out.emit_instr_reg_reg("    movd", xmm_regs[reg_idx], "eax");
                        self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                    } else {
                        self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[reg_idx], slot.0);
                    }
                }
                ParamClass::I128RegPair { base_reg_idx } => {
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx], slot.0);
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx + 1], slot.0 + 8);
                }
                ParamClass::StructByValReg { base_reg_idx, size } => {
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx], slot.0);
                    if size > 8 {
                        self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[base_reg_idx + 1], slot.0 + 8);
                    }
                }
                ParamClass::StructSseReg { lo_fp_idx, hi_fp_idx, .. } => {
                    self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[lo_fp_idx], slot.0);
                    if let Some(hi) = hi_fp_idx {
                        self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[hi], slot.0 + 8);
                    }
                }
                ParamClass::StructMixedIntSseReg { int_reg_idx, fp_reg_idx, .. } => {
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[int_reg_idx], slot.0);
                    self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[fp_reg_idx], slot.0 + 8);
                }
                ParamClass::StructMixedSseIntReg { fp_reg_idx, int_reg_idx, .. } => {
                    self.state.out.emit_instr_reg_rbp("    movq", xmm_regs[fp_reg_idx], slot.0);
                    self.state.out.emit_instr_reg_rbp("    movq", X86_ARG_REGS[int_reg_idx], slot.0 + 8);
                }
                ParamClass::F128AlwaysStack { offset } => {
                    let src = stack_base + offset;
                    self.state.out.emit_instr_rbp("    fldt", src);
                    self.state.out.emit_instr_rbp("    fstpt", slot.0);
                }
                ParamClass::I128Stack { offset } => {
                    let src = stack_base + offset;
                    self.state.out.emit_instr_rbp_reg("    movq", src, "rax");
                    self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                    self.state.out.emit_instr_rbp_reg("    movq", src + 8, "rax");
                    self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0 + 8);
                }
                ParamClass::StackScalar { offset } => {
                    // Load from caller's stack frame and store full 8 bytes to ensure
                    // the entire slot is initialized (see IntReg comment above).
                    let src = stack_base + offset;
                    self.state.out.emit_instr_rbp_reg("    movq", src, "rax");
                    self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                }
                ParamClass::StructStack { offset, size } | ParamClass::LargeStructStack { offset, size } => {
                    let src = stack_base + offset;
                    let n_qwords = size.div_ceil(8);
                    for qi in 0..n_qwords {
                        let src_off = src + (qi as i64 * 8);
                        let dst_off = slot.0 + (qi as i64 * 8);
                        self.state.out.emit_instr_rbp_reg("    movq", src_off, "rax");
                        self.state.out.emit_instr_reg_rbp("    movq", "rax", dst_off);
                    }
                }
                ParamClass::F128FpReg { .. } | ParamClass::F128GpPair { .. } | ParamClass::F128Stack { .. } |
                ParamClass::LargeStructByRefReg { .. } | ParamClass::LargeStructByRefStack { .. } |
                ParamClass::StructSplitRegStack { .. } |
                ParamClass::ZeroSizeSkip => {}
            }
        }
    }

    pub(super) fn emit_param_ref_impl(&mut self, dest: &Value, param_idx: usize, ty: IrType) {
        if param_idx >= self.state.param_classes.len() {
            return;
        }

        // If this param was pre-stored directly to its register-allocated
        // destination during emit_store_params, the value is already in place.
        // No code needs to be emitted — the register already holds the value.
        if self.state.param_pre_stored.contains(&param_idx) {
            return;
        }

        if param_idx < self.state.param_alloca_slots.len() {
            if let Some((slot, alloca_ty)) = self.state.param_alloca_slots[param_idx] {
                let load_instr = Self::mov_load_for_type(alloca_ty);
                let reg = Self::load_dest_reg(alloca_ty);
                let sr = self.slot_ref(slot.0);
                self.state.emit_fmt(format_args!("    {} {}, {}", load_instr, sr, reg));
                self.store_rax_to(dest);
                return;
            }
        }

        let xmm_regs = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"];
        let class = self.state.param_classes[param_idx];
        let stack_base: i64 = if self.state.omit_frame_pointer { 8 } else { 16 };

        match class {
            ParamClass::IntReg { reg_idx } => {
                let src_reg = Self::reg_for_type(X86_ARG_REGS[reg_idx], ty);
                let load_instr = Self::mov_load_for_type(ty);
                let dest_reg = Self::load_dest_reg(ty);
                self.state.emit_fmt(format_args!("    {} %{}, {}", load_instr, src_reg, dest_reg));
                self.store_rax_to(dest);
            }
            ParamClass::FloatReg { reg_idx } => {
                if ty == IrType::F32 {
                    self.state.out.emit_instr_reg_reg("    movd", xmm_regs[reg_idx], "eax");
                    self.store_rax_to(dest);
                } else {
                    self.state.out.emit_instr_reg_reg("    movq", xmm_regs[reg_idx], "rax");
                    self.store_rax_to(dest);
                }
            }
            ParamClass::StackScalar { offset } => {
                let src = stack_base + offset;
                let load_instr = Self::mov_load_for_type(ty);
                let reg = Self::load_dest_reg(ty);
                let sr = self.slot_ref(src);
                self.state.emit_fmt(format_args!("    {} {}, {}", load_instr, sr, reg));
                self.store_rax_to(dest);
            }
            _ => {}
        }
    }

    pub(super) fn emit_epilogue_and_ret_impl(&mut self, frame_size: i64) {
        self.emit_epilogue_impl(frame_size);
        if self.state.function_return_thunk {
            self.state.emit("    jmp __x86_return_thunk");
        } else {
            self.state.emit("    ret");
        }
    }

    pub(super) fn store_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        Self::mov_store_for_type(ty)
    }

    pub(super) fn load_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        Self::mov_load_for_type(ty)
    }
}
