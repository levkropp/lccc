//! X86Codegen: function call operations.

use crate::ir::reexports::{IrConst, Operand, Value};
use crate::common::types::IrType;
use crate::backend::call_abi::{CallAbiConfig, CallArgClass, compute_stack_push_bytes};
use crate::backend::generation::is_i128_type;
use super::emit::{X86Codegen, X86_ARG_REGS};

impl X86Codegen {
    pub(super) fn call_abi_config_impl(&self) -> CallAbiConfig {
        CallAbiConfig {
            max_int_regs: 6, max_float_regs: 8,
            align_i128_pairs: false,
            f128_in_fp_regs: false, f128_in_gp_pairs: false,
            variadic_floats_in_gp: false,
            large_struct_by_ref: false,
            use_sysv_struct_classification: true,
            use_riscv_float_struct_classification: false,
            allow_struct_split_reg_stack: false,
            align_struct_pairs: false,
            sret_uses_dedicated_reg: false,
        }
    }

    pub(super) fn emit_call_compute_stack_space_impl(&self, arg_classes: &[CallArgClass], _arg_types: &[IrType]) -> usize {
        compute_stack_push_bytes(arg_classes)
    }

    pub(super) fn emit_call_stack_args_impl(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                            _arg_types: &[IrType], stack_arg_space: usize, _fptr_spill: usize, _f128_temp_space: usize) -> i64 {
        let need_align_pad = !stack_arg_space.is_multiple_of(16);
        let mut sp_adjust: i64 = 0;
        if need_align_pad {
            self.state.emit("    subq $8, %rsp");
            sp_adjust += 8;
            // Adjust RSP frame size so operand_to_rax slot conversions
            // are correct after the alignment subq.
            if self.state.out.use_rsp_addressing {
                self.state.out.rsp_frame_size += 8;
            }
        }
        let arg_padding = crate::backend::call_abi::compute_stack_arg_padding(arg_classes);
        let stack_indices: Vec<usize> = (0..args.len())
            .filter(|&i| arg_classes[i].is_stack())
            .collect();
        for &si in stack_indices.iter().rev() {
            match arg_classes[si] {
                CallArgClass::F128Stack => {
                    match &args[si] {
                        Operand::Const(ref c) => {
                            let x87_bytes: [u8; 10] = match c {
                                IrConst::LongDouble(_, f128_bytes) => {
                                    let x87 = crate::common::long_double::f128_bytes_to_x87_bytes(f128_bytes);
                                    let mut b = [0u8; 10];
                                    b.copy_from_slice(&x87[..10]);
                                    b
                                }
                                _ => {
                                    let f64_val = c.to_f64().unwrap_or(0.0);
                                    crate::ir::reexports::f64_to_x87_bytes(f64_val)
                                }
                            };
                            let lo = u64::from_le_bytes(x87_bytes[0..8].try_into().unwrap());
                            let hi_2bytes = u16::from_le_bytes(x87_bytes[8..10].try_into().unwrap());
                            self.state.out.emit_instr_imm("    pushq", hi_2bytes as i64);
                            self.state.out.emit_instr_imm_reg("    movabsq", lo as i64, "rax");
                            self.state.emit("    pushq %rax");
                            self.state.reg_cache.invalidate_all();
                        }
                        Operand::Value(ref v) => {
                            if self.state.f128_direct_slots.contains(&v.0) {
                                if let Some(slot) = self.state.get_slot(v.0) {
                                    self.state.emit("    subq $16, %rsp");
                                    self.state.out.emit_instr_rbp("    fldt", slot.0);
                                    self.state.emit("    fstpt (%rsp)");
                                } else {
                                    self.state.emit("    subq $16, %rsp");
                                }
                            } else if let Some(slot) = self.state.get_slot(v.0) {
                                if self.state.is_alloca(v.0) {
                                    self.state.emit("    subq $16, %rsp");
                                    self.state.out.emit_instr_rbp("    fldt", slot.0);
                                    self.state.emit("    fstpt (%rsp)");
                                } else {
                                    self.state.out.emit_instr_rbp_reg("    movq", slot.0, "rax");
                                    self.state.reg_cache.invalidate_all();
                                    self.state.emit("    subq $16, %rsp");
                                    self.state.emit("    pushq %rax");
                                    self.state.emit("    fldl (%rsp)");
                                    self.state.emit("    addq $8, %rsp");
                                    self.state.emit("    fstpt (%rsp)");
                                }
                            } else {
                                self.state.emit("    subq $16, %rsp");
                            }
                            self.state.reg_cache.invalidate_all();
                        }
                    }
                }
                CallArgClass::I128Stack => {
                    // Push 128-bit value to stack. Load directly from slot
                    // to avoid operand_to_rax_rdx clobbering rdx.
                    match &args[si] {
                        Operand::Value(v) => {
                            if let Some(slot) = self.state.get_slot(v.0) {
                                let sr1 = self.slot_ref(slot.0 + 8);
                                let sr0 = self.slot_ref(slot.0);
                                self.state.emit_fmt(format_args!("    pushq {}", sr1));
                                self.state.emit_fmt(format_args!("    pushq {}", sr0));
                            } else {
                                self.state.emit("    pushq $0");
                                self.state.emit("    pushq $0");
                            }
                        }
                        Operand::Const(c) => {
                            if let IrConst::I128(v) = c {
                                let low = *v as u64;
                                let high = (*v >> 64) as u64;
                                self.state.emit_fmt(format_args!("    pushq ${}", high as i64));
                                self.state.emit_fmt(format_args!("    pushq ${}", low as i64));
                            } else {
                                // Smaller constant or zero
                                if let Operand::Value(_) = &args[si] {} // can't happen
                                self.operand_to_rax(&args[si]);
                                self.state.emit("    pushq $0");
                                self.state.emit("    pushq %rax");
                            }
                        }
                    }
                }
                CallArgClass::StructByValStack { size } | CallArgClass::LargeStructStack { size } => {
                    self.operand_to_rax(&args[si]);
                    let n_qwords = size.div_ceil(8);
                    for qi in (0..n_qwords).rev() {
                        let offset = qi * 8;
                        if offset + 8 <= size {
                            self.state.emit_fmt(format_args!("    pushq {}(%rax)", offset));
                        } else {
                            self.state.out.emit_instr_mem_reg("    movq", offset as i64, "rax", "rcx");
                            self.state.emit("    pushq %rcx");
                        }
                    }
                }
                CallArgClass::Stack => {
                    self.operand_to_rax(&args[si]);
                    self.state.emit("    pushq %rax");
                    if self.state.out.use_rsp_addressing {
                        self.state.out.rsp_frame_size += 8;
                    }
                    sp_adjust += 8;
                }
                _ => {}
            }
            let pad = arg_padding[si];
            if pad > 0 {
                self.state.out.emit_instr_imm_reg("    subq", pad as i64, "rsp");
                sp_adjust += pad as i64;
                if self.state.out.use_rsp_addressing {
                    self.state.out.rsp_frame_size += pad as i64;
                }
            }
        }
        // Restore the original RSP frame size. The total adjustment is
        // tracked in sp_adjust and returned to emit_call_reg_args.
        if self.state.out.use_rsp_addressing {
            self.state.out.rsp_frame_size -= sp_adjust;
        }
        // Return the total RSP adjustment so emit_call_reg_args can
        // compensate stack slot offsets when loading register arguments.
        sp_adjust
    }

    pub(super) fn emit_call_reg_args_impl(&mut self, args: &[Operand], arg_classes: &[CallArgClass],
                          _arg_types: &[IrType], total_sp_adjust: i64, _f128_temp_space: usize, _stack_arg_space: usize,
                          _struct_arg_riscv_float_classes: &[Option<crate::common::types::RiscvFloatClass>]) {
        // Stack args (Phase 2) may have adjusted rsp. Temporarily increase
        // the RSP frame size so rbp-to-rsp offset conversion is correct
        // when loading register arguments from stack slots.
        if total_sp_adjust != 0 && self.state.out.use_rsp_addressing {
            self.state.out.rsp_frame_size += total_sp_adjust;
        }
        let xmm_regs = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"];
        let mut float_count = 0usize;
        for (i, arg) in args.iter().enumerate() {
            match arg_classes[i] {
                CallArgClass::I128RegPair { base_reg_idx } => {
                    let lo_reg = X86_ARG_REGS[base_reg_idx];
                    let hi_reg = X86_ARG_REGS[base_reg_idx + 1];
                    // Load 128-bit value directly into the target register pair,
                    // avoiding operand_to_rax_rdx which clobbers rax and rdx
                    // (potentially overwriting previously-assigned arguments).
                    match arg {
                        Operand::Value(v) => {
                            if let Some(slot) = self.state.get_slot(v.0) {
                                // Load both halves directly from the stack slot
                                self.state.out.emit_instr_rbp_reg("    movq", slot.0, lo_reg);
                                self.state.out.emit_instr_rbp_reg("    movq", slot.0 + 8, hi_reg);
                            } else {
                                // No slot: zero both halves
                                self.state.emit_fmt(format_args!("    xorq %{}, %{}", lo_reg, lo_reg));
                                self.state.emit_fmt(format_args!("    xorq %{}, %{}", hi_reg, hi_reg));
                            }
                        }
                        Operand::Const(c) => {
                            match c {
                                IrConst::I128(v) => {
                                    let low = *v as u64;
                                    let high = (*v >> 64) as u64;
                                    self.state.emit_fmt(format_args!("    movabsq ${}, %{}", low as i64, lo_reg));
                                    self.state.emit_fmt(format_args!("    movabsq ${}, %{}", high as i64, hi_reg));
                                }
                                IrConst::Zero => {
                                    self.state.emit_fmt(format_args!("    xorq %{}, %{}", lo_reg, lo_reg));
                                    self.state.emit_fmt(format_args!("    xorq %{}, %{}", hi_reg, hi_reg));
                                }
                                _ => {
                                    // Smaller constant: load into lo_reg via rax, zero hi_reg
                                    self.operand_to_rax(arg);
                                    self.state.out.emit_instr_reg_reg("    movq", "rax", lo_reg);
                                    self.state.emit_fmt(format_args!("    xorq %{}, %{}", hi_reg, hi_reg));
                                }
                            }
                        }
                    }
                }
                CallArgClass::StructByValReg { base_reg_idx, size } => {
                    self.operand_to_rax(arg);
                    let lo_reg = X86_ARG_REGS[base_reg_idx];
                    self.state.out.emit_instr_mem_reg("    movq", 0, "rax", lo_reg);
                    if size > 8 {
                        let hi_reg = X86_ARG_REGS[base_reg_idx + 1];
                        self.state.out.emit_instr_mem_reg("    movq", 8, "rax", hi_reg);
                    }
                }
                CallArgClass::StructSseReg { lo_fp_idx, hi_fp_idx, .. } => {
                    self.operand_to_rax(arg);
                    self.state.out.emit_instr_mem_reg("    movq", 0, "rax", xmm_regs[lo_fp_idx]);
                    float_count += 1;
                    if let Some(hi) = hi_fp_idx {
                        self.state.out.emit_instr_mem_reg("    movq", 8, "rax", xmm_regs[hi]);
                        float_count += 1;
                    }
                }
                CallArgClass::StructMixedIntSseReg { int_reg_idx, fp_reg_idx, .. } => {
                    self.operand_to_rax(arg);
                    self.state.out.emit_instr_mem_reg("    movq", 8, "rax", xmm_regs[fp_reg_idx]);
                    float_count += 1;
                    self.state.out.emit_instr_mem_reg("    movq", 0, "rax", X86_ARG_REGS[int_reg_idx]);
                }
                CallArgClass::StructMixedSseIntReg { fp_reg_idx, int_reg_idx, .. } => {
                    self.operand_to_rax(arg);
                    self.state.out.emit_instr_mem_reg("    movq", 8, "rax", X86_ARG_REGS[int_reg_idx]);
                    self.state.out.emit_instr_mem_reg("    movq", 0, "rax", xmm_regs[fp_reg_idx]);
                    float_count += 1;
                }
                CallArgClass::FloatReg { reg_idx } => {
                    self.operand_to_rax(arg);
                    self.state.out.emit_instr_reg_reg("    movq", "rax", xmm_regs[reg_idx]);
                    float_count += 1;
                }
                CallArgClass::IntReg { reg_idx } => {
                    let target_reg = X86_ARG_REGS[reg_idx];
                    // Register-direct for constants: skip rax for zero and small imm
                    let did_direct = if let Operand::Const(c) = arg {
                        if let Some(imm) = c.to_i64() {
                            if imm == 0 {
                                let target_32 = match target_reg {
                                    "rdi" => "edi", "rsi" => "esi", "rdx" => "edx",
                                    "rcx" => "ecx", "r8" => "r8d", "r9" => "r9d",
                                    _ => target_reg,
                                };
                                self.state.emit_fmt(format_args!("    xorl %{}, %{}", target_32, target_32));
                            } else {
                                self.state.out.emit_instr_imm_reg("    movq", imm, target_reg);
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if !did_direct {
                        self.operand_to_rax(arg);
                        self.state.out.emit_instr_reg_reg("    movq", "rax", target_reg);
                    }
                }
                _ => {}
            }
        }
        if float_count > 0 {
            self.state.out.emit_instr_imm_reg("    movb", float_count as i64, "al");
        } else {
            self.state.emit("    xorl %eax, %eax");
        }
        self.state.reg_cache.invalidate_all();
        // Restore the original RSP frame size.
        if total_sp_adjust != 0 && self.state.out.use_rsp_addressing {
            self.state.out.rsp_frame_size -= total_sp_adjust;
        }
    }

    pub(super) fn emit_call_instruction_impl(&mut self, direct_name: Option<&str>, func_ptr: Option<&Operand>, _indirect: bool, _stack_arg_space: usize) {
        if let Some(name) = direct_name {
            if self.state.needs_plt(name) {
                self.state.emit_fmt(format_args!("    call {}@PLT", name));
            } else {
                self.state.out.emit_call(name);
            }
        } else if let Some(ptr) = func_ptr {
            // Load function pointer into rax and call through it.
            // Using rax avoids clobbering r10 which may hold a
            // register-allocated value.
            self.operand_to_rax(ptr);
            if self.state.indirect_branch_thunk {
                self.state.emit("    movq %rax, %r10");
                self.state.emit("    call __x86_indirect_thunk_r10");
            } else {
                self.state.emit("    call *%rax");
            }
        }
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_call_cleanup_impl(&mut self, stack_arg_space: usize, _f128_temp_space: usize, _indirect: bool) {
        let need_align_pad = !stack_arg_space.is_multiple_of(16);
        let total_cleanup = stack_arg_space + if need_align_pad { 8 } else { 0 };
        if total_cleanup > 0 {
            self.state.out.emit_instr_imm_reg("    addq", total_cleanup as i64, "rsp");
        }
    }

    pub(super) fn set_call_ret_eightbyte_classes_impl(&mut self, classes: &[crate::common::types::EightbyteClass]) {
        self.call_ret_classes = classes.to_vec();
    }

    pub(super) fn emit_call_store_result_impl(&mut self, dest: &Value, return_type: IrType) {
        if is_i128_type(return_type) {
            use crate::common::types::EightbyteClass;
            if self.call_ret_classes.len() == 2 {
                let (c0, c1) = (self.call_ret_classes[0], self.call_ret_classes[1]);
                match (c0, c1) {
                    (EightbyteClass::Integer, EightbyteClass::Sse) => {
                        if let Some(slot) = self.state.get_slot(dest.0) {
                            self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                            self.state.emit("    movq %xmm0, %rdx");
                            self.state.out.emit_instr_reg_rbp("    movq", "rdx", slot.0 + 8);
                        }
                        self.state.reg_cache.invalidate_all();
                    }
                    (EightbyteClass::Sse, EightbyteClass::Integer) => {
                        if let Some(slot) = self.state.get_slot(dest.0) {
                            self.state.emit("    movq %xmm0, %rdx");
                            self.state.out.emit_instr_reg_rbp("    movq", "rdx", slot.0);
                            self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0 + 8);
                        }
                        self.state.reg_cache.invalidate_all();
                    }
                    (EightbyteClass::Sse, EightbyteClass::Sse) => {
                        if let Some(slot) = self.state.get_slot(dest.0) {
                            self.state.emit("    movq %xmm0, %rax");
                            self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0);
                            self.state.emit("    movq %xmm1, %rax");
                            self.state.out.emit_instr_reg_rbp("    movq", "rax", slot.0 + 8);
                        }
                        self.state.reg_cache.invalidate_all();
                    }
                    _ => {
                        self.store_rax_rdx_to(dest);
                    }
                }
            } else {
                self.store_rax_rdx_to(dest);
            }
        } else if return_type == IrType::F32 {
            self.state.emit("    movd %xmm0, %eax");
            self.store_rax_to(dest);
        } else if return_type == IrType::F128 {
            if let Some(slot) = self.state.get_slot(dest.0) {
                self.state.out.emit_instr_rbp("    fstpt", slot.0);
                self.state.out.emit_instr_rbp("    fldt", slot.0);
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.state.reg_cache.set_acc(dest.0, false);
                self.state.f128_direct_slots.insert(dest.0);
            } else {
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.store_rax_to(dest);
            }
        } else if return_type == IrType::F64 {
            self.state.emit("    movq %xmm0, %rax");
            self.store_rax_to(dest);
        } else {
            self.store_rax_to(dest);
        }
    }

    pub(super) fn emit_call_store_i128_result_impl(&mut self, dest: &Value) {
        self.store_rax_rdx_to(dest);
    }

    pub(super) fn emit_call_move_f32_to_acc_impl(&mut self) {
        self.state.emit("    movd %xmm0, %eax");
    }

    pub(super) fn emit_call_move_f64_to_acc_impl(&mut self) {
        self.state.emit("    movq %xmm0, %rax");
    }
}
