//! Instruction selection: lower IR instructions to MachInst with virtual registers.
//!
//! Each IR instruction type has a lowering function that produces a sequence of
//! MachInst entries. Virtual registers (MachReg::Vreg) represent IR values that
//! will be assigned physical registers by the MachInst register allocator.
//! Pre-colored registers (MachReg::Phys) are used for x86 constraints like
//! division (rax:rdx) and shifts (rcx/%cl).

use crate::ir::reexports::{IrBinOp, Operand, Value, IrConst};
use crate::common::types::IrType;
use super::machinst::*;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Convert an IR Operand to a MachOperand.
fn lower_operand(op: &Operand) -> MachOperand {
    match op {
        Operand::Value(v) => MachOperand::Reg(MachReg::Vreg(v.0)),
        Operand::Const(c) => MachOperand::Imm(const_to_i64(c)),
    }
}

/// Convert an IrConst to an i64 value.
fn const_to_i64(c: &IrConst) -> i64 {
    match c {
        IrConst::I8(v) => *v as i64,
        IrConst::I16(v) => *v as i64,
        IrConst::I32(v) => *v as i64,
        IrConst::I64(v) => *v,
        IrConst::Zero => 0,
        // Float/i128/LongDouble constants: use bit representation
        IrConst::F32(v) => v.to_bits() as i64,
        IrConst::F64(v) => v.to_bits() as i64,
        IrConst::LongDouble(v, _) => v.to_bits() as i64,
        IrConst::I128(v) => *v as i64, // truncate to low 64 bits
    }
}

/// Check if an operand is an immediate that fits in a signed 32-bit value.
fn const_as_imm32(op: &Operand) -> Option<i64> {
    match op {
        Operand::Const(c) => {
            let v = const_to_i64(c);
            if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                Some(v)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Emit a move from an IR Operand to a virtual register.
fn emit_mov_operand(op: &Operand, dst: MachReg, size: OpSize, out: &mut Vec<MachInst>) {
    match op {
        Operand::Value(v) => {
            let src_reg = MachReg::Vreg(v.0);
            if src_reg != dst {
                out.push(MachInst::Mov {
                    src: MachOperand::Reg(src_reg),
                    dst: MachOperand::Reg(dst),
                    size,
                });
            }
        }
        Operand::Const(c) => {
            let val = const_to_i64(c);
            out.push(MachInst::Mov {
                src: MachOperand::Imm(val),
                dst: MachOperand::Reg(dst),
                size,
            });
        }
    }
}

/// Emit an ALU instruction with an IR Operand as source.
fn emit_alu_operand(op: AluOp, src: &Operand, dst: MachReg, size: OpSize, out: &mut Vec<MachInst>) {
    out.push(MachInst::Alu {
        op,
        src: lower_operand(src),
        dst,
        size,
    });
}

/// Map IrBinOp to AluOp for simple two-address operations.
fn binop_to_alu(op: IrBinOp) -> Option<AluOp> {
    match op {
        IrBinOp::Add => Some(AluOp::Add),
        IrBinOp::Sub => Some(AluOp::Sub),
        IrBinOp::And => Some(AluOp::And),
        IrBinOp::Or => Some(AluOp::Or),
        IrBinOp::Xor => Some(AluOp::Xor),
        IrBinOp::Mul => Some(AluOp::Imul),
        _ => None,
    }
}

/// Map IrBinOp to ShiftOp.
fn binop_to_shift(op: IrBinOp) -> Option<ShiftOp> {
    match op {
        IrBinOp::Shl => Some(ShiftOp::Shl),
        IrBinOp::LShr => Some(ShiftOp::Shr),
        IrBinOp::AShr => Some(ShiftOp::Sar),
        _ => None,
    }
}

/// Check if an immediate is a LEA scale factor (3, 5, or 9).
/// Returns the scale (2, 4, or 8) for the LEA index.
fn lea_scale_for_mul(imm: i64) -> Option<u8> {
    match imm {
        3 => Some(2),  // lea (%r, %r, 2), %r  → r * 3
        5 => Some(4),  // lea (%r, %r, 4), %r  → r * 5
        9 => Some(8),  // lea (%r, %r, 8), %r  → r * 9
        _ => None,
    }
}

// ── BinOp Lowering ───────────────────────────────────────────────────────

/// Lower an IR BinOp instruction to MachInst sequence.
///
/// Handles all 13 IrBinOp variants:
/// - Simple ALU (Add, Sub, And, Or, Xor, Mul): two-address form
/// - Shifts (Shl, AShr, LShr): count in %cl or immediate
/// - Division (SDiv, UDiv, SRem, URem): implicit rax:rdx pair
pub fn lower_binop(
    dest: &Value,
    op: IrBinOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let dst = MachReg::Vreg(dest.0);

    // ── Simple ALU operations (two-address form) ─────────────────────
    if let Some(alu_op) = binop_to_alu(op) {
        // Special case: multiply by LEA scale factor (3, 5, 9)
        if op == IrBinOp::Mul {
            if let Some(imm) = const_as_imm32(rhs) {
                if let Some(scale) = lea_scale_for_mul(imm) {
                    // lea (%dst, %dst, scale), %dst  →  dst * (scale+1)
                    emit_mov_operand(lhs, dst, size, out);
                    out.push(MachInst::Lea {
                        base: dst,
                        index: Some((dst, scale)),
                        offset: 0,
                        dst,
                    });
                    return;
                }
                // 3-operand imul: imul $imm, %src, %dst
                if imm != 0 && imm != 1 {
                    let src = match lhs {
                        Operand::Value(v) => MachReg::Vreg(v.0),
                        Operand::Const(_) => {
                            // Need to materialize const into dst first
                            emit_mov_operand(lhs, dst, size, out);
                            dst
                        }
                    };
                    out.push(MachInst::Imul3 { imm, src, dst, size });
                    return;
                }
            }
        }

        // General two-address: dst = lhs; dst OP= rhs
        emit_mov_operand(lhs, dst, size, out);
        emit_alu_operand(alu_op, rhs, dst, size, out);
        return;
    }

    // ── Shift operations ─────────────────────────────────────────────
    if let Some(shift_op) = binop_to_shift(op) {
        emit_mov_operand(lhs, dst, size, out);

        if let Some(imm) = const_as_imm32(rhs) {
            // Immediate shift: shlq $N, %dst
            let mask = if size == OpSize::S32 { 31 } else { 63 };
            out.push(MachInst::Shift {
                op: shift_op,
                amount: MachOperand::Imm(imm & mask),
                dst,
                size,
            });
        } else {
            // Variable shift: count must be in %cl (pre-color to rcx)
            emit_mov_operand(rhs, MachReg::Phys(RCX), size, out);
            out.push(MachInst::Shift {
                op: shift_op,
                amount: MachOperand::Reg(MachReg::Phys(RCX)),
                dst,
                size,
            });
        }
        return;
    }

    // ── Division and remainder ───────────────────────────────────────
    // x86 division uses implicit rax:rdx pair.
    // Signed: cqto (sign-extend rax → rdx:rax), then idivq.
    // Unsigned: xorl %edx, %edx (zero rdx), then divq.
    // Quotient → rax, remainder → rdx.
    match op {
        IrBinOp::SDiv | IrBinOp::SRem => {
            // Load dividend to rax (pre-colored)
            emit_mov_operand(lhs, MachReg::Phys(RAX), size, out);
            // Sign-extend rax → rdx:rax
            out.push(MachInst::Cqto { size });
            // Load divisor — must NOT be rax or rdx.
            // Use a vreg; the allocator will assign a non-rax/rdx register.
            // But if rhs is already a vreg, use it directly as the divisor operand.
            let divisor_op = lower_operand(rhs);
            out.push(MachInst::Div { divisor: divisor_op, signed: true, size });
            // Move result to dest vreg
            let result_phys = if op == IrBinOp::SDiv { RAX } else { RDX };
            out.push(MachInst::Mov {
                src: MachOperand::Reg(MachReg::Phys(result_phys)),
                dst: MachOperand::Reg(dst),
                size,
            });
        }
        IrBinOp::UDiv | IrBinOp::URem => {
            // Load dividend to rax
            emit_mov_operand(lhs, MachReg::Phys(RAX), size, out);
            // Zero-extend: xorl %edx, %edx
            out.push(MachInst::XorRdx);
            // Load divisor and divide
            let divisor_op = lower_operand(rhs);
            out.push(MachInst::Div { divisor: divisor_op, signed: false, size });
            // Move result to dest
            let result_phys = if op == IrBinOp::UDiv { RAX } else { RDX };
            out.push(MachInst::Mov {
                src: MachOperand::Reg(MachReg::Phys(result_phys)),
                dst: MachOperand::Reg(dst),
                size,
            });
        }
        _ => unreachable!("unhandled binop: {:?}", op),
    }
}
