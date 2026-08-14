//! Recognition of common integer bit-manipulation idioms.
//!
//! Source code often spells operations such as population count using portable
//! shifts and masks.  Recognizing the complete data-flow graph here lets every
//! backend select its native instruction without making codegen source-specific.

use crate::common::types::IrType;
use crate::ir::reexports::{Instruction, IrBinOp, IrCmpOp, IrConst, IrFunction, IrUnaryOp, Operand};

fn const_u64(op: Operand) -> Option<u64> {
    match op {
        Operand::Const(IrConst::I8(v)) => Some(v as u8 as u64),
        Operand::Const(IrConst::I16(v)) => Some(v as u16 as u64),
        Operand::Const(IrConst::I32(v)) => Some(v as u32 as u64),
        Operand::Const(IrConst::I64(v)) => Some(v as u64),
        Operand::Const(IrConst::Zero) => Some(0),
        _ => None,
    }
}

fn peel(mut op: Operand, defs: &[Option<Instruction>]) -> Operand {
    // Casts between the IR's integer carrier types are common after inlining.
    // The masks below constrain the recognized computation to 32 bits.
    for _ in 0..32 {
        let Operand::Value(v) = op else { break };
        match defs.get(v.0 as usize).and_then(Option::as_ref) {
            Some(Instruction::Cast { src, from_ty, to_ty, .. })
                if from_ty.is_integer() && to_ty.is_integer() => op = *src,
            Some(Instruction::Copy { src, .. }) => op = *src,
            _ => break,
        }
    }
    op
}

fn same(a: Operand, b: Operand, defs: &[Option<Instruction>]) -> bool {
    match (peel(a, defs), peel(b, defs)) {
        (Operand::Value(a), Operand::Value(b)) => a == b,
        (Operand::Const(a), Operand::Const(b)) => a.to_i128() == b.to_i128(),
        _ => false,
    }
}

fn binop(opnd: Operand, wanted: IrBinOp, defs: &[Option<Instruction>]) -> Option<(Operand, Operand, IrType)> {
    let Operand::Value(v) = peel(opnd, defs) else { return None };
    match defs.get(v.0 as usize).and_then(Option::as_ref) {
        Some(Instruction::BinOp { op, lhs, rhs, ty, .. }) if *op == wanted => Some((*lhs, *rhs, *ty)),
        _ => None,
    }
}

fn commutative_const(opnd: Operand, wanted: IrBinOp, constant: u64, defs: &[Option<Instruction>]) -> Option<Operand> {
    let (lhs, rhs, _) = binop(opnd, wanted, defs)?;
    if const_u64(rhs) == Some(constant) { Some(lhs) }
    else if const_u64(lhs) == Some(constant) { Some(rhs) }
    else { None }
}

fn shift(opnd: Operand, wanted: IrBinOp, amount: u64, defs: &[Option<Instruction>]) -> Option<Operand> {
    let (lhs, rhs, _) = binop(opnd, wanted, defs)?;
    (const_u64(rhs) == Some(amount)).then_some(lhs)
}

fn select(opnd: Operand, defs: &[Option<Instruction>]) -> Option<(Operand, Operand, Operand, IrType)> {
    let Operand::Value(v) = peel(opnd, defs) else { return None };
    match defs.get(v.0 as usize).and_then(Option::as_ref) {
        Some(Instruction::Select { cond, true_val, false_val, ty, .. }) =>
            Some((*cond, *true_val, *false_val, *ty)),
        _ => None,
    }
}

fn unsigned_le_const(opnd: Operand, constant: u64, defs: &[Option<Instruction>]) -> Option<Operand> {
    let Operand::Value(v) = peel(opnd, defs) else { return None };
    match defs.get(v.0 as usize).and_then(Option::as_ref) {
        Some(Instruction::Cmp { op: IrCmpOp::Ule, lhs, rhs, .. })
            if const_u64(*rhs) == Some(constant) => Some(*lhs),
        _ => None,
    }
}

fn add_const(opnd: Operand, amount: u64, defs: &[Option<Instruction>]) -> Option<Operand> {
    commutative_const(opnd, IrBinOp::Add, amount, defs)
}

fn is_incremented(true_val: Operand, false_val: Operand, amount: u64, defs: &[Option<Instruction>]) -> bool {
    if add_const(true_val, amount, defs).is_some_and(|base| same(base, false_val, defs)) {
        return true;
    }
    match (const_u64(peel(true_val, defs)), const_u64(peel(false_val, defs))) {
        (Some(a), Some(b)) => a == b.wrapping_add(amount),
        _ => false,
    }
}

/// Match the canonical 32-bit parallel population-count expression:
///
/// `x -= (x >> 1) & 0x55555555; ...; (x * 0x01010101) >> 24`
fn match_popcount32(result: Operand, defs: &[Option<Instruction>]) -> Option<Operand> {
    let multiplied = shift(result, IrBinOp::LShr, 24, defs)?;
    let stage3 = commutative_const(multiplied, IrBinOp::Mul, 0x0101_0101, defs)?;
    let stage3_add = commutative_const(stage3, IrBinOp::And, 0x0f0f_0f0f, defs)?;
    let (a, b, _) = binop(stage3_add, IrBinOp::Add, defs)?;
    let stage2 = if shift(a, IrBinOp::LShr, 4, defs).is_some_and(|base| same(base, b, defs)) { b }
        else if shift(b, IrBinOp::LShr, 4, defs).is_some_and(|base| same(base, a, defs)) { a }
        else { return None };

    let (a, b, _) = binop(stage2, IrBinOp::Add, defs)?;
    let a_base = commutative_const(a, IrBinOp::And, 0x3333_3333, defs)?;
    let b_base = commutative_const(b, IrBinOp::And, 0x3333_3333, defs)?;
    let stage1 = if shift(a_base, IrBinOp::LShr, 2, defs).is_some_and(|base| same(base, b_base, defs)) { b_base }
        else if shift(b_base, IrBinOp::LShr, 2, defs).is_some_and(|base| same(base, a_base, defs)) { a_base }
        else { return None };

    let (original, subtracted, ty) = binop(stage1, IrBinOp::Sub, defs)?;
    if ty != IrType::U32 && ty != IrType::I32 { return None; }
    let shifted = commutative_const(subtracted, IrBinOp::And, 0x5555_5555, defs)?;
    let shifted_base = shift(shifted, IrBinOp::LShr, 1, defs)?;
    same(original, shifted_base, defs).then_some(peel(original, defs))
}

/// Match the common binary-search implementation of 32-bit count-leading-zeros.
/// If-conversion turns each source-level `if` into paired selects: one shifts the
/// working value and the other increments the count.  Requiring both chains to
/// agree makes this substantially stricter than merely matching the constants.
fn match_clz32(result: Operand, defs: &[Option<Instruction>]) -> Option<Operand> {
    let (final_cond, final_true, final_false, final_ty) = select(result, defs)?;
    if final_ty != IrType::I32 && final_ty != IrType::U32 { return None; }
    let mut count = final_false;
    if !is_incremented(final_true, count, 1, defs) {
        return None;
    }
    let mut working = unsigned_le_const(final_cond, 0x7fff_ffff, defs)?;

    for (amount, threshold) in [
        (2, 0x3fff_ffff),
        (4, 0x0fff_ffff),
        (8, 0x00ff_ffff),
        (16, 0x0000_ffff),
    ] {
        let (count_cond, count_true, count_false, _) = select(count, defs)?;
        if !is_incremented(count_true, count_false, amount, defs) {
            return None;
        }
        let (value_cond, value_true, value_false, value_ty) = select(working, defs)?;
        if value_ty != IrType::U32 && value_ty != IrType::I32 { return None; }
        if !same(count_cond, value_cond, defs) {
            return None;
        }
        if !shift(value_true, IrBinOp::Shl, amount, defs).is_some_and(|base| same(base, value_false, defs)) {
            return None;
        }
        let compared = unsigned_le_const(value_cond, threshold, defs)?;
        if !same(compared, value_false, defs) { return None; }
        count = count_false;
        working = value_false;
    }
    (const_u64(peel(count, defs)) == Some(0)).then_some(peel(working, defs))
}

fn match_shift_pair(opnd: Operand, amount: u64, defs: &[Option<Instruction>]) -> Option<Operand> {
    let (a, b, _) = binop(opnd, IrBinOp::Or, defs)?;
    if let (Some(x), Some(y)) = (
        shift(a, IrBinOp::LShr, amount, defs),
        shift(b, IrBinOp::Shl, amount, defs),
    ) {
        if same(x, y, defs) { return Some(x); }
    }
    if let (Some(x), Some(y)) = (
        shift(b, IrBinOp::LShr, amount, defs),
        shift(a, IrBinOp::Shl, amount, defs),
    ) {
        if same(x, y, defs) { return Some(x); }
    }
    None
}

/// Match the final byte-swap portion of a mask-and-shift bit reversal:
///
/// `y = ((x >> 8) & 0x00ff00ff) | ((x & 0x00ff00ff) << 8);`
/// `result = (y >> 16) | (y << 16);`
fn match_bswap32_network(result: Operand, defs: &[Option<Instruction>]) -> Option<Operand> {
    let byte_swapped = match_shift_pair(result, 16, defs)?;
    let (a, b, ty) = binop(byte_swapped, IrBinOp::Or, defs)?;
    if ty != IrType::U32 && ty != IrType::I32 { return None; }

    let match_halves = |right: Operand, left: Operand| -> Option<Operand> {
        let right_shifted = commutative_const(right, IrBinOp::And, 0x00ff_00ff, defs)?;
        let original = shift(right_shifted, IrBinOp::LShr, 8, defs)?;
        let left_masked = shift(left, IrBinOp::Shl, 8, defs)?;
        let left_original = commutative_const(left_masked, IrBinOp::And, 0x00ff_00ff, defs)?;
        same(original, left_original, defs).then_some(peel(original, defs))
    };
    match_halves(a, b).or_else(|| match_halves(b, a))
}

fn match_masked_swap_stage(result: Operand, amount: u64, mask: u64, defs: &[Option<Instruction>]) -> Option<Operand> {
    let (a, b, ty) = binop(result, IrBinOp::Or, defs)?;
    if ty != IrType::U32 && ty != IrType::I32 { return None; }
    let match_halves = |right: Operand, left: Operand| -> Option<Operand> {
        let shifted = commutative_const(right, IrBinOp::And, mask, defs)?;
        let original = shift(shifted, IrBinOp::LShr, amount, defs)?;
        let masked = shift(left, IrBinOp::Shl, amount, defs)?;
        let left_original = commutative_const(masked, IrBinOp::And, mask, defs)?;
        same(original, left_original, defs).then_some(peel(original, defs))
    };
    match_halves(a, b).or_else(|| match_halves(b, a))
}

fn match_bit_reverse32(result: Operand, defs: &[Option<Instruction>]) -> Option<Operand> {
    let mut value = match_bswap32_network(result, defs)?;
    for (amount, mask) in [(4, 0x0f0f_0f0f), (2, 0x3333_3333), (1, 0x5555_5555)] {
        value = match_masked_swap_stage(value, amount, mask, defs)?;
    }
    Some(peel(value, defs))
}

pub(crate) fn recognize_function(func: &mut IrFunction, enable_bit_reverse: bool) -> usize {
    let mut defs = vec![None; func.max_value_id() as usize + 1];
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                defs[dest.0 as usize] = Some(inst.clone());
            }
        }
    }

    let mut changes = 0;
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            match inst {
                Instruction::BinOp { dest, op: IrBinOp::LShr, ty, .. }
                    if *ty == IrType::U32 || *ty == IrType::I32 => {
                    let result = Operand::Value(*dest);
                    if let Some(src) = match_popcount32(result, &defs) {
                        *inst = Instruction::UnaryOp { dest: *dest, op: IrUnaryOp::Popcount, src, ty: IrType::U32 };
                        changes += 1;
                    }
                }
                Instruction::BinOp { dest, op: IrBinOp::Or, ty, .. }
                    if *ty == IrType::U32 || *ty == IrType::I32 => {
                    let result = Operand::Value(*dest);
                    if enable_bit_reverse {
                        if let Some(src) = match_bit_reverse32(result, &defs) {
                            *inst = Instruction::UnaryOp { dest: *dest, op: IrUnaryOp::BitReverse, src, ty: IrType::U32 };
                            changes += 1;
                            continue;
                        }
                    }
                    if let Some(src) = match_bswap32_network(result, &defs) {
                        *inst = Instruction::UnaryOp { dest: *dest, op: IrUnaryOp::Bswap, src, ty: IrType::U32 };
                        changes += 1;
                    }
                }
                Instruction::Select { dest, ty, .. }
                    if *ty == IrType::U32 || *ty == IrType::I32 => {
                    let result = Operand::Value(*dest);
                    if let Some(src) = match_clz32(result, &defs) {
                        *inst = Instruction::UnaryOp { dest: *dest, op: IrUnaryOp::Clz, src, ty: IrType::U32 };
                        changes += 1;
                    }
                }
                _ => {}
            }
        }
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::reexports::Value;

    fn swar_defs(second_mask: i64) -> Vec<Option<Instruction>> {
        let value = |id| Operand::Value(Value(id));
        let constant = |n| Operand::Const(IrConst::I64(n));
        let mut defs = vec![None; 13];
        let mut put = |id, op, lhs, rhs| {
            defs[id as usize] = Some(Instruction::BinOp {
                dest: Value(id), op, lhs, rhs, ty: IrType::U32,
            });
        };
        put(1, IrBinOp::LShr, value(0), constant(1));
        put(2, IrBinOp::And, value(1), constant(0x5555_5555));
        put(3, IrBinOp::Sub, value(0), value(2));
        put(4, IrBinOp::And, value(3), constant(0x3333_3333));
        put(5, IrBinOp::LShr, value(3), constant(2));
        put(6, IrBinOp::And, value(5), constant(second_mask));
        put(7, IrBinOp::Add, value(4), value(6));
        put(8, IrBinOp::LShr, value(7), constant(4));
        put(9, IrBinOp::Add, value(7), value(8));
        put(10, IrBinOp::And, value(9), constant(0x0f0f_0f0f));
        put(11, IrBinOp::Mul, value(10), constant(0x0101_0101));
        put(12, IrBinOp::LShr, value(11), constant(24));
        defs
    }

    #[test]
    fn recognizes_canonical_swar_popcount32() {
        let defs = swar_defs(0x3333_3333);
        assert!(same(match_popcount32(Operand::Value(Value(12)), &defs).unwrap(), Operand::Value(Value(0)), &defs));
    }

    #[test]
    fn rejects_near_miss_swar_popcount32() {
        let defs = swar_defs(0x3333_3331);
        assert!(match_popcount32(Operand::Value(Value(12)), &defs).is_none());
    }
}
