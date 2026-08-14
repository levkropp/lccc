//! X86Codegen: floating-point binary operations and F128 negation.

use crate::ir::reexports::{IrUnaryOp, IrConst, Operand, Value};
use crate::backend::cast::FloatOp;
use crate::common::types::IrType;
use super::emit::{phys_reg_name, phys_reg_name_32, typed_phys_reg_name, is_xmm_reg};
use super::emit::X86Codegen;

impl X86Codegen {
    pub(super) fn emit_float_binop_impl(&mut self, dest: &Value, op: FloatOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if ty == IrType::F128 {
            let x87_op = match op {
                FloatOp::Add => "faddp",
                FloatOp::Sub => "fsubrp",
                FloatOp::Mul => "fmulp",
                FloatOp::Div => "fdivrp",
            };
            self.emit_f128_load_to_x87(lhs);
            self.emit_f128_load_to_x87(rhs);
            self.state.emit_fmt(format_args!("    {} %st, %st(1)", x87_op));
            if let Some(dest_slot) = self.state.get_slot(dest.0) {
                self.state.out.emit_instr_rbp("    fstpt", dest_slot.0);
                self.state.out.emit_instr_rbp("    fldt", dest_slot.0);
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.state.reg_cache.set_acc(dest.0, false);
                self.state.f128_direct_slots.insert(dest.0);
            } else {
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.state.reg_cache.invalidate_acc();
                self.store_rax_to(dest);
            }
            return;
        }
        let mnemonic = self.emit_float_binop_mnemonic_impl(op);
        let suffix = if ty == IrType::F64 { "sd" } else { "ss" };

        // Load LHS → %xmm0 and RHS → %xmm1, keeping FP values in the SSE
        // domain (constant pool for literals, movsd/movss for XMM-allocated
        // values and stack slots; GPR round-trip only as a fallback).
        self.emit_fp_operand_to_xmm(lhs, ty, "xmm0");
        self.emit_fp_operand_to_xmm(rhs, ty, "xmm1");

        self.state.emit_fmt(format_args!("    {}{} %xmm1, %xmm0", mnemonic, suffix));
        self.store_xmm_to(dest, "xmm0", ty);
    }

    pub(super) fn emit_float_binop_impl_impl(&mut self, _mnemonic: &str, _ty: IrType) {
        unreachable!("x86 emit_float_binop_impl should not be called directly");
    }

    pub(super) fn emit_float_binop_mnemonic_impl(&self, op: FloatOp) -> &'static str {
        match op {
            FloatOp::Add => "add",
            FloatOp::Sub => "sub",
            FloatOp::Mul => "mul",
            FloatOp::Div => "div",
        }
    }

    pub(super) fn emit_unaryop_impl(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand, ty: IrType) {
        if ty == IrType::F128 && op == IrUnaryOp::Neg {
            self.emit_f128_load_to_x87(src);
            self.state.emit("    fchs");
            if let Some(dest_slot) = self.state.get_slot(dest.0) {
                self.state.out.emit_instr_rbp("    fstpt", dest_slot.0);
                self.state.out.emit_instr_rbp("    fldt", dest_slot.0);
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.state.reg_cache.set_acc(dest.0, false);
                self.state.f128_direct_slots.insert(dest.0);
            } else {
                self.state.emit("    subq $8, %rsp");
                self.state.emit("    fstpl (%rsp)");
                self.state.emit("    popq %rax");
                self.state.reg_cache.invalidate_acc();
                self.store_rax_to(dest);
            }
            return;
        }
        // Register-direct path for integer unary ops when dest has a register.
        if !ty.is_float() && !matches!(ty, IrType::I128 | IrType::U128 | IrType::F128) {
            if let Some(d_reg) = self.dest_reg(dest) {
                if !is_xmm_reg(d_reg) {
                    let use_32bit = matches!(ty, IrType::I32 | IrType::U32);
                    let d_name = if use_32bit { phys_reg_name_32(d_reg) } else { phys_reg_name(d_reg) };
                    let suffix = if use_32bit { "l" } else { "q" };

                    match op {
                        IrUnaryOp::Neg => {
                            self.operand_to_callee_reg(src, d_reg);
                            self.state.emit_fmt(format_args!("    neg{} %{}", suffix, d_name));
                            self.state.reg_cache.invalidate_acc();
                            return;
                        }
                        IrUnaryOp::Not => {
                            self.operand_to_callee_reg(src, d_reg);
                            self.state.emit_fmt(format_args!("    not{} %{}", suffix, d_name));
                            self.state.reg_cache.invalidate_acc();
                            return;
                        }
                        IrUnaryOp::Bswap => {
                            self.operand_to_callee_reg(src, d_reg);
                            if matches!(ty, IrType::I16 | IrType::U16) {
                                // bswap doesn't exist for 16-bit; use rolw with 16-bit register
                                let reg16 = typed_phys_reg_name(d_reg, ty);
                                self.state.emit_fmt(format_args!("    rolw $8, %{}", reg16));
                            } else {
                                let bswap_name = if use_32bit { phys_reg_name_32(d_reg) } else { phys_reg_name(d_reg) };
                                let bswap_suffix = if use_32bit { "l" } else { "q" };
                                self.state.emit_fmt(format_args!("    bswap{} %{}", bswap_suffix, bswap_name));
                            }
                            self.state.reg_cache.invalidate_acc();
                            return;
                        }
                        IrUnaryOp::Popcount => {
                            // popcnt is two-operand: popcnt %src, %dest
                            if let Some(s_reg) = self.operand_reg(src) {
                                if !is_xmm_reg(s_reg) {
                                    let s_name = if use_32bit { phys_reg_name_32(s_reg) } else { phys_reg_name(s_reg) };
                                    self.state.emit_fmt(format_args!("    popcnt{} %{}, %{}", suffix, s_name, d_name));
                                    self.state.reg_cache.invalidate_acc();
                                    return;
                                }
                            }
                            self.operand_to_callee_reg(src, d_reg);
                            self.state.emit_fmt(format_args!("    popcnt{} %{}, %{}", suffix, d_name, d_name));
                            self.state.reg_cache.invalidate_acc();
                            return;
                        }
                        _ => {} // Clz, Ctz — complex multi-instruction, fall through to default
                    }
                }
            }
        }

        crate::backend::traits::emit_unaryop_default(self, dest, op, src, ty);
    }

    /// Load an FP operand directly into an XMM register, using the constant pool
    /// for FP literal constants instead of going through a GPR. Value operands
    /// allocated to an XMM register or a stack slot are loaded directly with
    /// `movsd`/`movss`; the GPR path is only a last-resort fallback.
    pub(super) fn emit_fp_operand_to_xmm(&mut self, op: &Operand, ty: IrType, xmm: &str) {
        match op {
            Operand::Const(IrConst::F64(v)) => {
                let bits = v.to_bits();
                if bits == 0 {
                    self.state.emit_fmt(format_args!("    xorpd %{}, %{}", xmm, xmm));
                } else {
                    let label = self.state.get_fp_const_label(bits);
                    self.state.emit_fmt(format_args!("    movsd {}(%rip), %{}", label, xmm));
                }
            }
            Operand::Const(IrConst::F32(v)) => {
                let bits = v.to_bits() as u64;
                if bits == 0 {
                    self.state.emit_fmt(format_args!("    xorps %{}, %{}", xmm, xmm));
                } else {
                    let label = self.state.get_fp_const_label(bits);
                    self.state.emit_fmt(format_args!("    movss {}(%rip), %{}", label, xmm));
                }
            }
            _ => {
                // Value operand: keep FP values in the SSE domain.
                if let Operand::Value(v) = op {
                    if let Some(&reg) = self.reg_assignments.get(&v.0) {
                        let reg_name = phys_reg_name(reg);
                        if is_xmm_reg(reg) {
                            // XMM → XMM: movapd copies all 128 bits. The
                            // register-register movsd form would preserve the
                            // destination's upper 64 bits, creating a false
                            // dependency on the old destination value that
                            // serializes against in-flight divsd/sqrtsd chains
                            // (measured ~2.4x slowdown on nbody under Rosetta).
                            if reg_name != xmm {
                                self.state.emit_fmt(format_args!("    movapd %{}, %{}", reg_name, xmm));
                            }
                            return;
                        }
                        // GPR → XMM (bit-reinterpreted value)
                        if ty == IrType::F32 {
                            self.state.emit_fmt(format_args!("    movd %{}, %{}", phys_reg_name_32(reg), xmm));
                        } else {
                            self.state.emit_fmt(format_args!("    movq %{}, %{}", reg_name, xmm));
                        }
                        return;
                    }
                    if !self.state.is_alloca(v.0) && !self.state.vector_values.contains(&v.0) {
                        if let Some(slot) = self.state.get_slot(v.0) {
                            // Stack slot → XMM: direct load, no GPR round-trip.
                            let instr = if ty == IrType::F32 { "    movss" } else { "    movsd" };
                            self.state.out.emit_instr_rbp_reg(instr, slot.0, xmm);
                            return;
                        }
                    }
                }
                // Fall back to GPR → XMM path
                let gpr = if xmm == "xmm0" { "rax" } else { "rcx" };
                let gpr32 = if xmm == "xmm0" { "eax" } else { "ecx" };
                if gpr == "rax" {
                    self.operand_to_rax(op);
                } else {
                    self.operand_to_rcx(op);
                }
                if ty == IrType::F32 {
                    self.state.emit_fmt(format_args!("    movd %{}, %{}", gpr32, xmm));
                } else {
                    self.state.emit_fmt(format_args!("    movq %{}, %{}", gpr, xmm));
                }
            }
        }
    }

    /// Stage an FP store value in an XMM register: returns the value's own
    /// XMM register when it has one, otherwise loads it into %xmm0 and
    /// returns "xmm0". Never clobbers %rcx (safe after address setup).
    pub(super) fn fp_store_value_xmm(&mut self, val: &Operand, ty: IrType) -> &'static str {
        if let Operand::Value(v) = val {
            if let Some(&reg) = self.reg_assignments.get(&v.0) {
                if is_xmm_reg(reg) {
                    return phys_reg_name(reg);
                }
            }
        }
        self.emit_fp_operand_to_xmm(val, ty, "xmm0");
        "xmm0"
    }
}
