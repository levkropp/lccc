//! ArmCodegen: floating-point binary operations.

use crate::ir::reexports::{IrConst, Operand, Value};
use crate::common::types::IrType;
use crate::backend::cast::FloatOp;
use crate::backend::traits::ArchCodegen;
use super::emit::{ArmCodegen, arm_fp_name, is_arm_fp_phys};

/// Return the AArch64 FMOV-immediate imm8 field when an IEEE-754 constant is
/// representable by the architecture's compact floating-point expansion.
fn fmov_imm8(bits: u64, ty: IrType) -> Option<u8> {
    for imm in 0u16..=255 {
        let imm = imm as u8;
        let sign = (imm >> 7) as u64;
        let b = ((imm >> 6) & 1) as u64;
        let cd = ((imm >> 4) & 3) as u64;
        let frac = (imm & 0xf) as u64;
        let expanded = match ty {
            IrType::F64 => {
                let exponent = ((b ^ 1) << 10)
                    | ((if b != 0 { 0xff } else { 0 }) << 2)
                    | cd;
                (sign << 63) | (exponent << 52) | (frac << 48)
            }
            IrType::F32 => {
                let exponent = ((b ^ 1) << 7)
                    | ((if b != 0 { 0x1f } else { 0 }) << 2)
                    | cd;
                (sign << 31) | (exponent << 23) | (frac << 19)
            }
            _ => return None,
        };
        if expanded == bits {
            return Some(imm);
        }
    }
    None
}

fn fp_target_reg(target: &str, ty: IrType) -> Option<u32> {
    let prefix = if ty == IrType::F32 { 's' } else { 'd' };
    let rest = target.strip_prefix(prefix)?;
    rest.parse::<u32>().ok().filter(|&reg| reg <= 31)
}

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
                } else if let (Some(imm8), Some(reg)) =
                    (fmov_imm8(bits, ty), fp_target_reg(target, ty))
                {
                    // The integrated assembler does not yet parse decimal FP
                    // immediates, so emit the architecturally exact instruction
                    // word. FMOV (scalar, immediate): imm8 at bits 20:13 and
                    // Rd at 4:0; type selects the S/D base opcode.
                    let base = if ty == IrType::F32 {
                        0x1e20_1000u32
                    } else {
                        0x1e60_1000u32
                    };
                    let word = base | ((imm8 as u32) << 13) | reg;
                    self.state.emit_fmt(format_args!("    .inst 0x{:08x}", word));
                } else {
                    // The constant pool lives in .rodata; at link time it can
                    // be arbitrarily far from the code (sections concatenate),
                    // so a ±1MB PC-relative literal load (`ldr dN, .LCFP_x`)
                    // truncates in large programs — SQLite's object hit
                    // R_AARCH64_LD_PREL_LO19 overflows. Use the
                    // range-unlimited adrp+ldr form.
                    let label = self.state.get_fp_const_label(bits);
                    self.state.emit_fmt(format_args!("    adrp x0, {}", label));
                    self.state.emit_fmt(format_args!("    ldr {}, [x0, :lo12:{}]", target, label));
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

    /// Fused multiply-subtract: `sub_dest = acc - mul_lhs*mul_rhs` (fmsub)
    /// when the multiply was the Sub's RHS, or `sub_dest = mul_lhs*mul_rhs - acc`
    /// (fnmsub: -acc + product) when it was the LHS.
    pub(super) fn emit_fused_mul_sub_impl(
        &mut self, _mul_dest: &Value, mul_lhs: &Operand, mul_rhs: &Operand,
        acc: &Operand, sub_dest: &Value, ty: IrType, mul_is_lhs: bool,
    ) {
        let (r0, r1, r2) = if ty == IrType::F32 { ("s0", "s1", "s2") } else { ("d0", "d1", "d2") };
        let acc_reg = self.float_operand_reg(acc, ty, r2);
        let lhs_reg = self.float_operand_reg(mul_lhs, ty, r0);
        let rhs_reg = self.float_operand_reg(mul_rhs, ty, r1);
        let output = self.reg_assignments.get(&sub_dest.0).copied()
            .filter(|r| is_arm_fp_phys(*r)).map(|r| arm_fp_name(r, ty))
            .unwrap_or_else(|| r0.to_string());
        let mnemonic = if mul_is_lhs { "fnmsub" } else { "fmsub" };
        self.state.emit_fmt(format_args!("    {} {}, {}, {}, {}", mnemonic, output, lhs_reg, rhs_reg, acc_reg));
        self.store_float_reg(sub_dest, ty, &output);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmov_immediate_encoding_matches_aarch64() {
        assert_eq!(fmov_imm8(1.0f64.to_bits(), IrType::F64), Some(0x70));
        assert_eq!(
            0x1e60_1000u32 | ((fmov_imm8(1.0f64.to_bits(), IrType::F64).unwrap() as u32) << 13),
            0x1e6e_1000,
        );
        assert_eq!(
            0x1e60_1000u32
                | ((fmov_imm8((-2.0f64).to_bits(), IrType::F64).unwrap() as u32) << 13)
                | 3,
            0x1e70_1003,
        );
        assert_eq!(
            0x1e20_1000u32
                | ((fmov_imm8(0.5f32.to_bits() as u64, IrType::F32).unwrap() as u32) << 13)
                | 4,
            0x1e2c_1004,
        );
        assert_eq!(fmov_imm8(0.1f64.to_bits(), IrType::F64), None);
    }
}
