//! X86Codegen: comparison and select operations.

use crate::ir::reexports::{BlockId, IrCmpOp, IrConst, Operand, Value};
use crate::common::types::IrType;
use super::emit::{X86Codegen, phys_reg_name, phys_reg_name_32, is_xmm_reg};

impl X86Codegen {
    pub(super) fn emit_float_cmp_impl(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        let swap_operands = matches!(op, IrCmpOp::Slt | IrCmpOp::Ult | IrCmpOp::Sle | IrCmpOp::Ule);
        let (first, second) = if swap_operands { (rhs, lhs) } else { (lhs, rhs) };

        // Load first operand → %xmm0 (use constant pool for FP constants)
        self.emit_fp_operand_to_xmm(first, ty, "xmm0");
        // Load second operand → %xmm1
        self.emit_fp_operand_to_xmm(second, ty, "xmm1");

        if ty == IrType::F64 {
            self.state.emit("    ucomisd %xmm1, %xmm0");
        } else {
            self.state.emit("    ucomiss %xmm1, %xmm0");
        }
        match op {
            IrCmpOp::Eq => {
                self.state.emit("    setnp %al");
                self.state.emit("    sete %cl");
                self.state.emit("    andb %cl, %al");
            }
            IrCmpOp::Ne => {
                self.state.emit("    setp %al");
                self.state.emit("    setne %cl");
                self.state.emit("    orb %cl, %al");
            }
            IrCmpOp::Slt | IrCmpOp::Ult | IrCmpOp::Sgt | IrCmpOp::Ugt => {
                self.state.emit("    seta %al");
            }
            IrCmpOp::Sle | IrCmpOp::Ule | IrCmpOp::Sge | IrCmpOp::Uge => {
                self.state.emit("    setae %al");
            }
        }
        self.state.emit("    movzbl %al, %eax");
        self.state.reg_cache.invalidate_acc();
        self.store_rax_to(dest);
    }

    pub(super) fn emit_f128_cmp_impl(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand) {
        let swap_x87 = matches!(op, IrCmpOp::Slt | IrCmpOp::Ult | IrCmpOp::Sle | IrCmpOp::Ule);
        let (first_x87, second_x87) = if swap_x87 { (lhs, rhs) } else { (rhs, lhs) };
        self.emit_f128_load_to_x87(first_x87);
        self.emit_f128_load_to_x87(second_x87);
        self.state.emit("    fucomip %st(1), %st");
        self.state.emit("    fstp %st(0)");
        match op {
            IrCmpOp::Eq => {
                self.state.emit("    setnp %al");
                self.state.emit("    sete %cl");
                self.state.emit("    andb %cl, %al");
            }
            IrCmpOp::Ne => {
                self.state.emit("    setp %al");
                self.state.emit("    setne %cl");
                self.state.emit("    orb %cl, %al");
            }
            IrCmpOp::Slt | IrCmpOp::Ult | IrCmpOp::Sgt | IrCmpOp::Ugt => {
                self.state.emit("    seta %al");
            }
            IrCmpOp::Sle | IrCmpOp::Ule | IrCmpOp::Sge | IrCmpOp::Uge => {
                self.state.emit("    setae %al");
            }
        }
        self.state.emit("    movzbl %al, %eax");
        self.state.reg_cache.invalidate_acc();
        self.store_rax_to(dest);
    }

    pub(super) fn emit_int_cmp_impl(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        let use_32bit = ty == IrType::I32 || ty == IrType::U32;
        self.emit_int_cmp_insn_typed(lhs, rhs, use_32bit);

        let set_instr = match op {
            IrCmpOp::Eq => "sete",
            IrCmpOp::Ne => "setne",
            IrCmpOp::Slt => "setl",
            IrCmpOp::Sle => "setle",
            IrCmpOp::Sgt => "setg",
            IrCmpOp::Sge => "setge",
            IrCmpOp::Ult => "setb",
            IrCmpOp::Ule => "setbe",
            IrCmpOp::Ugt => "seta",
            IrCmpOp::Uge => "setae",
        };
        self.state.emit_fmt(format_args!("    {} %al", set_instr));

        // Register-direct: movzbl %al, %dest_reg_32 — skip %rax relay.
        // Safe because %al is part of %rax, never overlaps callee-saved registers.
        if let Some(d_reg) = self.dest_reg(dest) {
            if !is_xmm_reg(d_reg) {
                let d_name = phys_reg_name_32(d_reg);
                self.state.emit_fmt(format_args!("    movzbl %al, %{}", d_name));
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }

        self.state.emit("    movzbl %al, %eax");
        self.state.reg_cache.invalidate_acc();
        self.store_rax_to(dest);
    }

    pub(super) fn emit_fused_cmp_branch_impl(
        &mut self,
        op: IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
        ty: IrType,
        true_label: &str,
        false_label: &str,
    ) {
        let use_32bit = ty == IrType::I32 || ty == IrType::U32;
        self.emit_int_cmp_insn_typed(lhs, rhs, use_32bit);

        let jcc = match op {
            IrCmpOp::Eq  => "je",
            IrCmpOp::Ne  => "jne",
            IrCmpOp::Slt => "jl",
            IrCmpOp::Sle => "jle",
            IrCmpOp::Sgt => "jg",
            IrCmpOp::Sge => "jge",
            IrCmpOp::Ult => "jb",
            IrCmpOp::Ule => "jbe",
            IrCmpOp::Ugt => "ja",
            IrCmpOp::Uge => "jae",
        };
        self.state.emit_fmt(format_args!("    {} {}", jcc, true_label));
        self.state.out.emit_jmp_label(false_label);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_fused_cmp_branch_blocks_impl(
        &mut self,
        op: IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
        ty: IrType,
        true_block: BlockId,
        false_block: BlockId,
    ) {
        let use_32bit = ty == IrType::I32 || ty == IrType::U32;
        self.emit_int_cmp_insn_typed(lhs, rhs, use_32bit);

        let jcc = match op {
            IrCmpOp::Eq  => "    je",
            IrCmpOp::Ne  => "    jne",
            IrCmpOp::Slt => "    jl",
            IrCmpOp::Sle => "    jle",
            IrCmpOp::Sgt => "    jg",
            IrCmpOp::Sge => "    jge",
            IrCmpOp::Ult => "    jb",
            IrCmpOp::Ule => "    jbe",
            IrCmpOp::Ugt => "    ja",
            IrCmpOp::Uge => "    jae",
        };
        self.state.out.emit_jcc_block(jcc, true_block.0);
        self.state.out.emit_jmp_block(false_block.0);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_cond_branch_blocks_impl(&mut self, cond: &Operand, true_block: BlockId, false_block: BlockId) {
        // Register-direct: test the condition register directly, skip %rax relay.
        if let Operand::Value(v) = cond {
            if let Some(&reg) = self.reg_assignments.get(&v.0) {
                if !is_xmm_reg(reg) {
                    let name = phys_reg_name(reg);
                    self.state.emit_fmt(format_args!("    testq %{}, %{}", name, name));
                    self.state.out.emit_jcc_block("    jne", true_block.0);
                    self.state.out.emit_jmp_block(false_block.0);
                    return;
                }
            }
        }
        self.operand_to_rax(cond);
        self.state.emit("    testq %rax, %rax");
        self.state.out.emit_jcc_block("    jne", true_block.0);
        self.state.out.emit_jmp_block(false_block.0);
    }

    pub(super) fn emit_select_impl(&mut self, dest: &Value, cond: &Operand, true_val: &Operand, false_val: &Operand, _ty: IrType) {
        // Register-direct: when dest has a register, operate directly on it.
        if let Some(d_reg) = self.dest_reg(dest) {
            if !is_xmm_reg(d_reg) {
                let d_name = phys_reg_name(d_reg);

                // Check for register conflicts: if true_val is in dest reg,
                // we must load it to %rcx BEFORE loading false_val to dest.
                let true_in_dest = match true_val {
                    Operand::Value(v) => self.reg_assignments.get(&v.0).copied() == Some(d_reg),
                    _ => false,
                };

                // Safe approach: use push/pop to save true_val on the stack,
                // avoiding register clobbering between operand loads.
                // The register allocator may assign overlapping registers to
                // cond, true_val, and false_val since they're all "live" at the
                // Select point but the allocator doesn't model simultaneous liveness.

                // Step 1: load false_val to dest
                self.operand_to_callee_reg(false_val, d_reg);

                // Step 2: load true_val to rax, push it (safe from clobbering)
                self.operand_to_rax(true_val);
                self.state.emit("    pushq %rax");

                // Step 3: load and test condition
                self.operand_to_rax(cond);
                self.state.emit("    testq %rax, %rax");

                // Step 4: pop true_val to rcx and cmov
                self.state.emit("    popq %rcx");
                self.state.emit_fmt(format_args!("    cmovneq %rcx, %{}", d_name));
                self.state.reg_cache.invalidate_acc();
                return;
            }
        }

        // Accumulator fallback
        self.operand_to_rax(false_val);
        self.operand_to_rcx(true_val);

        // Use %r11 for the condition when %rdx is allocated to a value.
        let cond_reg = if self.reg_assignments.values().any(|r| r.0 == 16) { "r11" } else { "rdx" };
        let cond_reg_32 = if cond_reg == "r11" { "r11d" } else { "edx" };

        match cond {
            Operand::Const(c) => {
                let val = match c {
                    IrConst::I8(v) => *v as i64,
                    IrConst::I16(v) => *v as i64,
                    IrConst::I32(v) => *v as i64,
                    IrConst::I64(v) => *v,
                    IrConst::Zero => 0,
                    _ => 0,
                };
                if val == 0 {
                    self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", cond_reg_32));
                } else if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
                    self.state.out.emit_instr_imm_reg("    movq", val, cond_reg);
                } else {
                    self.state.out.emit_instr_imm_reg("    movabsq", val, cond_reg);
                }
            }
            Operand::Value(v) => {
                if let Some(&reg) = self.reg_assignments.get(&v.0) {
                    let reg_name = phys_reg_name(reg);
                    self.state.out.emit_instr_reg_reg("    movq", reg_name, cond_reg);
                } else if self.state.get_slot(v.0).is_some() {
                    self.value_to_reg(v, cond_reg);
                } else {
                    self.state.emit_fmt(format_args!("    xorl %{0}, %{0}", cond_reg_32));
                }
            }
        }
        self.state.emit_fmt(format_args!("    testq %{0}, %{0}", cond_reg));
        self.state.emit("    cmovneq %rcx, %rax");
        self.state.reg_cache.invalidate_acc();
        self.store_rax_to(dest);
    }
}
