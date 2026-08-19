//! ArmCodegen: floating-point binary operations.

use crate::ir::reexports::{IrConst, Operand, Value};
use crate::common::types::IrType;
use crate::backend::cast::FloatOp;
use crate::backend::traits::ArchCodegen;
use super::emit::{ArmCodegen, arm_fp_name, is_arm_fp_phys};

impl ArmCodegen {
    pub(super) fn float_operand_reg(&mut self, op: &Operand, ty: IrType, scratch: &str) -> String {
        if let Operand::Value(v) = op {
            if let Some(&phys) = self.reg_assignments.get(&v.0) {
                if is_arm_fp_phys(phys) {
                    return arm_fp_name(phys, ty);
                }
            }
        }
        self.float_operand_to_reg(op, ty, scratch);
        scratch.to_string()
    }

    pub(super) fn float_operand_to_reg(&mut self, op: &Operand, ty: IrType, target: &str) {
        if let Operand::Value(v) = op {
            if let Some(&phys) = self.reg_assignments.get(&v.0) {
                if is_arm_fp_phys(phys) {
                    let source = arm_fp_name(phys, ty);
                    if source != target {
                        self.state.emit_fmt(format_args!("    fmov {}, {}", target, source));
                    }
                    return;
                }
            }
            // Stack-homed FP value: load straight into the FP register
            // (`ldr dN, [slot]`) instead of bouncing through a GPR
            // (`ldr x0` + `fmov dN, x0` = 2 instructions plus transfer latency).
            if !self.state.is_alloca(v.0) {
                if let Some(slot) = self.state.get_slot(v.0) {
                    self.emit_load_from_sp(target, slot.0, "ldr");
                    return;
                }
            }
        }
        // FP constants load straight from the .rodata constant pool via a
        // PC-relative literal load (1 instruction vs. a 5-instruction
        // movz/movk + fmov sequence).
        if let Operand::Const(c) = op {
            let bits = match (c, ty) {
                (IrConst::F64(v), IrType::F64) => Some(v.to_bits()),
                (IrConst::F32(v), IrType::F32) => Some(v.to_bits() as u64),
                _ => None,
            };
            if let Some(bits) = bits {
                if bits == 0 {
                    let zero = if ty == IrType::F32 { "wzr" } else { "xzr" };
                    self.state.emit_fmt(format_args!("    fmov {}, {}", target, zero));
                } else {
                    let label = self.state.get_fp_const_label(bits);
                    self.state.emit_fmt(format_args!("    ldr {}, {}", target, label));
                }
                return;
            }
        }
        self.operand_to_x0(op);
        if ty == IrType::F32 {
            self.state.emit_fmt(format_args!("    fmov {}, w0", target));
        } else {
            self.state.emit_fmt(format_args!("    fmov {}, x0", target));
        }
    }

    pub(super) fn store_float_reg(&mut self, dest: &Value, ty: IrType, source: &str) {
        if let Some(&phys) = self.reg_assignments.get(&dest.0) {
            if is_arm_fp_phys(phys) {
                let target = arm_fp_name(phys, ty);
                if target != source {
                    self.state.emit_fmt(format_args!("    fmov {}, {}", target, source));
                }
                return;
            }
        }
        // Stack-homed dest with a slot: store the FP register directly
        // (`str dN, [slot]`) instead of round-tripping through x0
        // (`fmov x0, dN; str x0, [slot]`).
        if !self.state.is_alloca(dest.0) {
            if let Some(slot) = self.state.get_slot(dest.0) {
                self.emit_store_to_sp(source, slot.0, "str");
                return;
            }
        }
        if ty == IrType::F32 {
            self.state.emit_fmt(format_args!("    fmov w0, {}", source));
        } else {
            self.state.emit_fmt(format_args!("    fmov x0, {}", source));
        }
        self.store_x0_to(dest);
    }

    pub(super) fn emit_fused_mul_add_impl(
        &mut self, _mul_dest: &Value, mul_lhs: &Operand, mul_rhs: &Operand,
        acc: &Operand, add_dest: &Value, ty: IrType,
    ) {
        let (r0, r1, r2) = if ty == IrType::F32 { ("s0", "s1", "s2") } else { ("d0", "d1", "d2") };
        let acc_reg = self.float_operand_reg(acc, ty, r2);
        let lhs_reg = self.float_operand_reg(mul_lhs, ty, r0);
        let rhs_reg = self.float_operand_reg(mul_rhs, ty, r1);
        let output = self.reg_assignments.get(&add_dest.0).copied()
            .filter(|r| is_arm_fp_phys(*r)).map(|r| arm_fp_name(r, ty))
            .unwrap_or_else(|| r0.to_string());
        self.state.emit_fmt(format_args!("    fmadd {}, {}, {}, {}", output, lhs_reg, rhs_reg, acc_reg));
        self.store_float_reg(add_dest, ty, &output);
    }

    pub(super) fn emit_float_binop_impl(&mut self, dest: &Value, op: FloatOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if ty == IrType::F128 {
            crate::backend::f128_softfloat::f128_emit_binop(self, dest, op, lhs, rhs);
            return;
        }
        // Non-F128: use default path.
        let mnemonic = self.emit_float_binop_mnemonic(op);
        let scratch0 = if ty == IrType::F32 { "s0" } else { "d0" };
        let r1 = if ty == IrType::F32 { "s1" } else { "d1" };
        let output = self.reg_assignments.get(&dest.0).copied()
            .filter(|r| is_arm_fp_phys(*r))
            .map(|r| arm_fp_name(r, ty))
            .unwrap_or_else(|| scratch0.to_string());
        let lhs_reg = self.float_operand_reg(lhs, ty, scratch0);
        let rhs_reg = self.float_operand_reg(rhs, ty, r1);
        self.state.emit_fmt(format_args!("    {} {}, {}, {}", mnemonic, output, lhs_reg, rhs_reg));
        self.store_float_reg(dest, ty, &output);
    }

    pub(super) fn emit_float_binop_body(&mut self, mnemonic: &str, ty: IrType) {
        self.state.emit("    mov x2, x0");
        if ty == IrType::F32 {
            self.state.emit("    fmov s0, w1");
            self.state.emit("    fmov s1, w2");
            self.state.emit_fmt(format_args!("    {} s0, s0, s1", mnemonic));
            self.state.emit("    fmov w0, s0");
            self.state.emit("    mov w0, w0"); // zero-extend
        } else {
            self.state.emit("    fmov d0, x1");
            self.state.emit("    fmov d1, x2");
            self.state.emit_fmt(format_args!("    {} d0, d0, d1", mnemonic));
            self.state.emit("    fmov x0, d0");
        }
    }

    pub(super) fn emit_f128_neg_impl(&mut self, dest: &Value, src: &Operand) {
        self.emit_f128_neg_full(dest, src);
    }
}
