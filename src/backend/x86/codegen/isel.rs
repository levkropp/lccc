//! Instruction selection: lower IR instructions to MachInst with virtual registers.
//!
//! Each IR instruction type has a lowering function that produces a sequence of
//! MachInst entries. Virtual registers (MachReg::Vreg) represent IR values that
//! will be assigned physical registers by the MachInst register allocator.
//! Pre-colored registers (MachReg::Phys) are used for x86 constraints like
//! division (rax:rdx) and shifts (rcx/%cl).

use crate::ir::reexports::{IrBinOp, IrCmpOp, IrUnaryOp, Operand, Value, IrConst, Instruction, Terminator, BlockId};
use crate::common::types::IrType;
use crate::backend::regalloc::PhysReg;
use crate::common::fx_hash::FxHashMap;
use super::machinst::*;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Convert an IR Value to a MachReg, using its physical register if already
/// allocated by the main register allocator.
fn value_to_reg(v: &Value, reg_assignments: &FxHashMap<u32, PhysReg>) -> MachReg {
    if let Some(&phys) = reg_assignments.get(&v.0) {
        // XMM registers (20-25) are for floats — shouldn't appear in integer MachInst.
        // Treat them as vregs so they get spilled to stack (safe fallback).
        if phys.0 >= 20 {
            return MachReg::Vreg(v.0);
        }
        MachReg::Phys(phys)
    } else {
        MachReg::Vreg(v.0)
    }
}

/// Convert an IR Operand to a MachOperand, using physical registers for
/// values that already have register assignments from the main allocator.
fn lower_operand_with_regs(op: &Operand, reg_assignments: &FxHashMap<u32, PhysReg>) -> MachOperand {
    match op {
        Operand::Value(v) => MachOperand::Reg(value_to_reg(v, reg_assignments)),
        Operand::Const(c) => MachOperand::Imm(const_to_i64(c)),
    }
}

/// Convert an IR Operand to a MachOperand (without register lookup — for internal use).
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

/// Emit a move from an IR Operand to a register, using physical registers
/// for values already allocated by the main register allocator.
fn emit_mov_operand_r(op: &Operand, dst: MachReg, size: OpSize,
                      ra: &FxHashMap<u32, PhysReg>, out: &mut Vec<MachInst>) {
    match op {
        Operand::Value(v) => {
            let src_reg = value_to_reg(v, ra);
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

/// Emit a move (legacy wrapper without reg_assignments — for internal helpers).
fn emit_mov_operand(op: &Operand, dst: MachReg, size: OpSize, out: &mut Vec<MachInst>) {
    emit_mov_operand_r(op, dst, size, &FxHashMap::default(), out);
}

/// Emit an ALU instruction with an IR Operand as source.
/// For large immediates that don't fit in i32, materialize to rax first.
fn emit_alu_operand_r(op: AluOp, src: &Operand, dst: MachReg, size: OpSize,
                      ra: &FxHashMap<u32, PhysReg>, out: &mut Vec<MachInst>) {
    let src_op = lower_operand_with_regs(src, ra);
    // x86 ALU instructions only support i32 immediates. For larger values,
    // materialize to the scratch register (rax) first.
    if let MachOperand::Imm(v) = &src_op {
        if *v < i32::MIN as i64 || *v > i32::MAX as i64 {
            out.push(MachInst::Mov {
                src: MachOperand::Imm(*v),
                dst: MachOperand::Reg(MachReg::Phys(RAX)),
                size,
            });
            out.push(MachInst::Alu {
                op,
                src: MachOperand::Reg(MachReg::Phys(RAX)),
                dst,
                size,
            });
            return;
        }
    }
    out.push(MachInst::Alu { op, src: src_op, dst, size });
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
pub fn lower_binop(
    dest: &Value,
    op: IrBinOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let dst = value_to_reg(dest, ra);

    // ── Simple ALU operations (two-address form) ─────────────────────
    if let Some(alu_op) = binop_to_alu(op) {
        if op == IrBinOp::Mul {
            if let Some(imm) = const_as_imm32(rhs) {
                if let Some(scale) = lea_scale_for_mul(imm) {
                    emit_mov_operand_r(lhs, dst, size, ra, out);
                    out.push(MachInst::Lea {
                        base: dst, index: Some((dst, scale)), offset: 0, dst,
                    });
                    return;
                }
                if imm != 0 && imm != 1 {
                    let src = match lhs {
                        Operand::Value(v) => value_to_reg(v, ra),
                        Operand::Const(_) => {
                            emit_mov_operand_r(lhs, dst, size, ra, out);
                            dst
                        }
                    };
                    out.push(MachInst::Imul3 { imm, src, dst, size });
                    return;
                }
            }
        }
        emit_mov_operand_r(lhs, dst, size, ra, out);
        emit_alu_operand_r(alu_op, rhs, dst, size, ra, out);
        return;
    }

    // ── Shift operations ─────────────────────────────────────────────
    if let Some(shift_op) = binop_to_shift(op) {
        emit_mov_operand_r(lhs, dst, size, ra, out);
        if let Some(imm) = const_as_imm32(rhs) {
            let mask = if size == OpSize::S32 { 31 } else { 63 };
            out.push(MachInst::Shift {
                op: shift_op, amount: MachOperand::Imm(imm & mask), dst, size,
            });
        } else {
            emit_mov_operand_r(rhs, MachReg::Phys(RCX), size, ra, out);
            out.push(MachInst::Shift {
                op: shift_op, amount: MachOperand::Reg(MachReg::Phys(RCX)), dst, size,
            });
        }
        return;
    }

    // ── Division and remainder ───────────────────────────────────────
    match op {
        IrBinOp::SDiv | IrBinOp::SRem => {
            emit_mov_operand_r(lhs, MachReg::Phys(RAX), size, ra, out);
            out.push(MachInst::Cqto { size });
            let divisor_op = lower_operand_with_regs(rhs, ra);
            out.push(MachInst::Div { divisor: divisor_op, signed: true, size });
            let result_phys = if op == IrBinOp::SDiv { RAX } else { RDX };
            out.push(MachInst::Mov {
                src: MachOperand::Reg(MachReg::Phys(result_phys)),
                dst: MachOperand::Reg(dst), size,
            });
        }
        IrBinOp::UDiv | IrBinOp::URem => {
            emit_mov_operand_r(lhs, MachReg::Phys(RAX), size, ra, out);
            out.push(MachInst::XorRdx);
            let divisor_op = lower_operand_with_regs(rhs, ra);
            out.push(MachInst::Div { divisor: divisor_op, signed: false, size });
            let result_phys = if op == IrBinOp::UDiv { RAX } else { RDX };
            out.push(MachInst::Mov {
                src: MachOperand::Reg(MachReg::Phys(result_phys)),
                dst: MachOperand::Reg(dst), size,
            });
        }
        _ => unreachable!("unhandled binop: {:?}", op),
    }
}

// ── Load / Store / Copy ──────────────────────────────────────────────────

/// Lower an IR Load: dest = *ptr.
pub fn lower_load(
    dest: &Value,
    ptr: &Value,
    ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let dst = value_to_reg(dest, ra);
    let base = value_to_reg(ptr, ra);
    out.push(MachInst::Mov {
        src: MachOperand::Mem { base, offset: 0 },
        dst: MachOperand::Reg(dst),
        size,
    });
}

/// Lower an IR Store: *ptr = val.
pub fn lower_store(
    val: &Operand,
    ptr: &Value,
    ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let base = value_to_reg(ptr, ra);
    let src = lower_operand_with_regs(val, ra);
    out.push(MachInst::Mov {
        src,
        dst: MachOperand::Mem { base, offset: 0 },
        size,
    });
}

/// Lower an IR Copy: dest = src.
pub fn lower_copy(
    dest: &Value,
    src: &Operand,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let dst = value_to_reg(dest, ra);
    emit_mov_operand_r(src, dst, OpSize::S64, ra, out);
}

// ── Comparison ───────────────────────────────────────────────────────────

/// Map IrCmpOp to CondCode.
fn cmp_to_cc(op: IrCmpOp) -> CondCode {
    match op {
        IrCmpOp::Eq => CondCode::E,
        IrCmpOp::Ne => CondCode::Ne,
        IrCmpOp::Slt => CondCode::L,
        IrCmpOp::Sle => CondCode::Le,
        IrCmpOp::Sgt => CondCode::G,
        IrCmpOp::Sge => CondCode::Ge,
        IrCmpOp::Ult => CondCode::B,
        IrCmpOp::Ule => CondCode::Be,
        IrCmpOp::Ugt => CondCode::A,
        IrCmpOp::Uge => CondCode::Ae,
    }
}

/// Lower an IR Cmp: dest = lhs CMP rhs (boolean result).
pub fn lower_cmp(
    dest: &Value,
    op: IrCmpOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let dst = value_to_reg(dest, ra);
    let cc = cmp_to_cc(op);
    let lhs_op = lower_operand_with_regs(lhs, ra);
    let rhs_op = lower_operand_with_regs(rhs, ra);
    out.push(MachInst::Cmp { lhs: lhs_op, rhs: rhs_op, size });
    out.push(MachInst::SetCC { cc, dst });
    out.push(MachInst::Movzx { src: dst, dst, from_size: OpSize::S8, to_size: OpSize::S32 });
}

/// Lower a fused Cmp + CondBranch (no boolean materialization).
pub fn lower_cmp_branch(
    op: IrCmpOp,
    lhs: &Operand,
    rhs: &Operand,
    ty: IrType,
    true_block: BlockId,
    false_block: BlockId,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let cc = cmp_to_cc(op);
    let lhs_op = lower_operand(lhs);
    let rhs_op = lower_operand(rhs);
    out.push(MachInst::Cmp { lhs: lhs_op, rhs: rhs_op, size });
    out.push(MachInst::Jcc { cc, target: format!(".LBB{}", true_block.0) });
    out.push(MachInst::Jmp { target: format!(".LBB{}", false_block.0) });
}

// ── Cast ─────────────────────────────────────────────────────────────────

/// Lower an IR Cast (integer-to-integer only; float casts go through Raw).
pub fn lower_cast(
    dest: &Value,
    src: &Operand,
    from_ty: IrType,
    to_ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let dst = value_to_reg(dest, ra);
    let from_size = OpSize::from_ir_type(from_ty);
    let to_size = OpSize::from_ir_type(to_ty);

    if to_size as u8 <= from_size as u8 {
        emit_mov_operand_r(src, dst, to_size, ra, out);
        return;
    }

    let src_reg = match src {
        Operand::Value(v) => value_to_reg(v, ra),
        Operand::Const(_) => {
            emit_mov_operand_r(src, dst, to_size, ra, out);
            return;
        }
    };

    if to_ty.is_unsigned() || from_ty.is_unsigned() {
        out.push(MachInst::Movzx { src: src_reg, dst, from_size, to_size });
    } else {
        out.push(MachInst::Movsx { src: src_reg, dst, from_size, to_size });
    }
}

// ── Unary Operations ─────────────────────────────────────────────────────

/// Lower an IR UnaryOp (neg, not only; bswap/clz/ctz/popcount go through Raw).
pub fn lower_unaryop(
    dest: &Value,
    op: IrUnaryOp,
    src: &Operand,
    ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) -> bool {
    let size = OpSize::from_ir_type(ty);
    let dst = value_to_reg(dest, ra);

    match op {
        IrUnaryOp::Neg => {
            emit_mov_operand_r(src, dst, size, ra, out);
            out.push(MachInst::Neg { dst, size });
            true
        }
        IrUnaryOp::Not => {
            emit_mov_operand_r(src, dst, size, ra, out);
            out.push(MachInst::Not { dst, size });
            true
        }
        _ => false,
    }
}

// ── Select (conditional move) ────────────────────────────────────────────

/// Lower an IR Select: dest = cond ? true_val : false_val.
pub fn lower_select(
    dest: &Value,
    cond: &Operand,
    true_val: &Operand,
    false_val: &Operand,
    ty: IrType,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let size = OpSize::from_ir_type(ty);
    let dst = value_to_reg(dest, ra);
    emit_mov_operand_r(false_val, dst, size, ra, out);
    emit_mov_operand_r(true_val, MachReg::Phys(RAX), size, ra, out);
    let cond_op = lower_operand_with_regs(cond, ra);
    out.push(MachInst::Test { lhs: cond_op.clone(), rhs: cond_op, size: OpSize::S64 });
    out.push(MachInst::Cmov {
        cc: CondCode::Ne,
        src: MachOperand::Reg(MachReg::Phys(RAX)),
        dst, size,
    });
}

// ── GEP (pointer arithmetic) ─────────────────────────────────────────────

/// Lower an IR GetElementPtr: dest = base + offset.
pub fn lower_gep(
    dest: &Value,
    base: &Value,
    offset: &Operand,
    ra: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) {
    let dst = value_to_reg(dest, ra);
    let base_reg = value_to_reg(base, ra);

    if let Some(imm) = const_as_imm32(offset) {
        if imm == 0 {
            if base_reg != dst {
                out.push(MachInst::Mov {
                    src: MachOperand::Reg(base_reg),
                    dst: MachOperand::Reg(dst),
                    size: OpSize::S64,
                });
            }
        } else {
            out.push(MachInst::Lea { base: base_reg, index: None, offset: imm, dst });
        }
        return;
    }

    out.push(MachInst::Mov {
        src: MachOperand::Reg(base_reg),
        dst: MachOperand::Reg(dst),
        size: OpSize::S64,
    });
    out.push(MachInst::Alu {
        op: AluOp::Add,
        src: lower_operand_with_regs(offset, ra),
        dst,
        size: OpSize::S64,
    });
}

// ── Terminator lowering ──────────────────────────────────────────────────

/// Lower a conditional branch (non-fused): test cond, jne true, jmp false.
pub fn lower_cond_branch(
    cond: &Operand,
    true_block: BlockId,
    false_block: BlockId,
    out: &mut Vec<MachInst>,
) {
    let cond_op = lower_operand(cond);
    out.push(MachInst::Test { lhs: cond_op.clone(), rhs: cond_op, size: OpSize::S64 });
    out.push(MachInst::Jcc { cc: CondCode::Ne, target: format!(".LBB{}", true_block.0) });
    out.push(MachInst::Jmp { target: format!(".LBB{}", false_block.0) });
}

// ── Block-level lowering (integration entry point) ───────────────────────

/// Check if an IR instruction can be lowered to MachInst.
/// Instructions that can't are emitted as Raw passthrough via the existing codegen.
pub fn can_lower(inst: &Instruction) -> bool {
    match inst {
        Instruction::BinOp { ty, .. } => !ty.is_float() && !ty.is_128bit(),
        Instruction::Load { ty, .. } => !ty.is_float() && !ty.is_128bit() && !ty.is_long_double(),
        Instruction::Store { ty, .. } => !ty.is_float() && !ty.is_128bit() && !ty.is_long_double(),
        Instruction::Copy { .. } => true,
        Instruction::Cmp { ty, .. } => !ty.is_float() && !ty.is_128bit(),
        Instruction::Cast { from_ty, to_ty, .. } => {
            !from_ty.is_float() && !to_ty.is_float()
            && !from_ty.is_128bit() && !to_ty.is_128bit()
            && !from_ty.is_long_double() && !to_ty.is_long_double()
        }
        Instruction::UnaryOp { op, ty, .. } => {
            !ty.is_float() && !ty.is_128bit()
            && matches!(op, IrUnaryOp::Neg | IrUnaryOp::Not)
        }
        Instruction::Select { ty, .. } => !ty.is_float() && !ty.is_128bit(),
        Instruction::GetElementPtr { .. } => true,
        Instruction::GlobalAddr { .. } => true,
        _ => false,
    }
}

/// Lower a single IR instruction to MachInst.
/// Returns true if lowered, false if it should use Raw passthrough.
pub fn lower_instruction(
    inst: &Instruction,
    reg_assignments: &FxHashMap<u32, PhysReg>,
    out: &mut Vec<MachInst>,
) -> bool {
    // For values that already have register allocations from the existing
    // allocator, use their physical register directly (MachReg::Phys).
    // The MachInst allocator only handles the remaining Vreg values.

    let ra = reg_assignments;
    match inst {
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            if ty.is_float() || ty.is_128bit() { return false; }
            // Only handle I32/U32/I64/U64/Ptr — sub-32-bit types have complex
            // register sub-register interactions that need special handling.
            if matches!(ty, IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16) { return false; }
            lower_binop(dest, *op, lhs, rhs, *ty, ra, out);
            true
        }
        Instruction::Load { .. } => {
            // Load interactions with alloca/pointer addresses are complex — fall back.
            false
        }
        Instruction::Store { .. } => {
            // Store interactions with alloca addresses are complex — fall back to
            // existing codegen for now. Will be migrated in Phase 2f.
            false
        }
        Instruction::Copy { dest, src } => {
            lower_copy(dest, src, ra, out);
            true
        }
        Instruction::Cmp { .. } => {
            // Cmp lowering disabled: the existing codegen's SetCC+Movzbl path
            // has edge cases with register naming. Re-enable after fixing.
            false
        }
        Instruction::Cast { .. } => {
            // Cast lowering disabled: register naming edge cases with
            // sign/zero extension and 32-bit sub-registers.
            false
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            if ty.is_float() || ty.is_128bit() { return false; }
            if matches!(ty, IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16) { return false; }
            lower_unaryop(dest, *op, src, *ty, ra, out)
        }
        Instruction::Select { dest, cond, true_val, false_val, ty } => {
            if ty.is_float() || ty.is_128bit() { return false; }
            // cmov doesn't exist for 8-bit operands — fall back
            if matches!(ty, IrType::I8 | IrType::U8) { return false; }
            lower_select(dest, cond, true_val, false_val, *ty, ra, out);
            true
        }
        Instruction::GetElementPtr { .. } => {
            // GEP involves pointer addresses — fall back for now.
            false
        }
        _ => false,
    }
}
