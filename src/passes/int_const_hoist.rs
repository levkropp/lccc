//! Integer constant hoisting for loop bodies.
//!
//! Integer constants that do not fit the AArch64 add/cmp immediate forms
//! (imm12: 0..=4095, or cmn negative range) are otherwise materialized with
//! movz/movk inside the loop on every iteration — sieve's marking-loop bound
//! `cmp j, #10000000` cost two instructions per iteration. This pass
//! materializes each distinct large constant once in the loop preheader as a
//! `Copy`, making it a register-allocatable SSA value (the register steal
//! rebalances it if a hotter value needs the register more).
//!
//! Constants are collected from BinOp/Cmp operands (the forms whose immediate
//! encoding is range-limited); small constants and zero (`wzr`/`xzr`) are
//! already free. A hoisted value is reused by nested loops (the outer
//! preheader dominates them) but never across sibling loops.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::analysis;
use crate::ir::reexports::{Instruction, IrBinOp, IrConst, IrFunction, Operand, Value};
use super::loop_analysis;

/// Run integer-constant hoisting on a function. Returns the number of
/// distinct constants materialized in preheaders.
pub(crate) fn run(func: &mut IrFunction) -> usize {
    if func.blocks.len() < 2 {
        return 0;
    }
    let cfg = analysis::CfgAnalysis::build(func);
    let mut loops = loop_analysis::merge_loops_by_header(loop_analysis::find_natural_loops(
        cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom));
    if loops.is_empty() {
        return 0;
    }
    // Outermost first so constants used by nested loops hoist as far out as
    // possible and inner loops can reuse the dominating value.
    loops.sort_by_key(|l| l.body.len());
    loops.reverse();

    // const bits -> (hoisted value id, body of the loop whose preheader defines it)
    let mut hoisted: FxHashMap<u64, (u32, FxHashSet<usize>)> = FxHashMap::default();
    let mut count = 0;
    for lp in &loops {
        let Some(preheader) = loop_analysis::find_preheader(lp.header, &lp.body, &cfg.preds) else {
            continue;
        };
        // Distinct large constants used in this loop body.
        let mut consts: Vec<u64> = Vec::new();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        for &bi in &lp.body {
            for inst in &func.blocks[bi].instructions {
                for_each_int_operand(inst, &mut |op, needs_reg| {
                    if let Some(bits) = large_int_const(op, needs_reg) {
                        if seen.insert(bits) {
                            consts.push(bits);
                        }
                    }
                });
            }
        }

        for bits in consts {
            // Reuse a previously hoisted value when its definition dominates
            // this loop (i.e. this loop is nested inside the defining one).
            let reusable = hoisted.get(&bits).filter(|(_, body)| lp.body.is_subset(body));
            let new_val = if let Some(&(vid, _)) = reusable {
                vid
            } else {
                let vid = func.next_value_id;
                func.next_value_id += 1;
                func.blocks[preheader].instructions.push(Instruction::Copy {
                    dest: Value(vid),
                    src: Operand::Const(IrConst::I64(bits as i64)),
                });
                hoisted.insert(bits, (vid, lp.body.clone()));
                count += 1;
                vid
            };
            // Rewrite uses within the loop body.
            for &bi in &lp.body {
                for inst in &mut func.blocks[bi].instructions {
                    rewrite_int_operands(inst, bits, new_val);
                }
            }
        }
    }
    count
}

/// An integer constant that is NOT encodable as an AArch64 add/cmp immediate
/// (imm12 0..=4095, or the cmn negative range -4095..=-1) and so would be
/// materialized with movz/movk. Returns the value as u64 bits.
/// With `needs_reg` (mul/div/rem operands — those have no immediate form at
/// all), any constant qualifies: even `mov w0, #26` per iteration is waste.
fn large_int_const(op: &Operand, needs_reg: bool) -> Option<u64> {
    let v: i64 = match op {
        Operand::Const(IrConst::I8(v)) => *v as i64,
        Operand::Const(IrConst::I16(v)) => *v as i64,
        Operand::Const(IrConst::I32(v)) => *v as i64,
        Operand::Const(IrConst::I64(v)) => *v,
        _ => return None,
    };
    if !needs_reg && (-4095..=4095).contains(&v) {
        return None; // imm12 / cmn encodable — free already
    }
    Some(v as u64)
}

/// Visit operands of the instruction forms with range-limited immediate
/// encodings. The bool is true when the operand position has no immediate
/// encoding at all (mul/div/rem), so even small constants pay a materialization.
fn for_each_int_operand(inst: &Instruction, f: &mut dyn FnMut(&Operand, bool)) {
    match inst {
        Instruction::BinOp { op, lhs, rhs, .. } => {
            let needs_reg = matches!(
                op,
                IrBinOp::Mul | IrBinOp::SDiv | IrBinOp::UDiv | IrBinOp::SRem | IrBinOp::URem
            );
            f(lhs, needs_reg);
            f(rhs, needs_reg);
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            f(lhs, false);
            f(rhs, false);
        }
        _ => {}
    }
}

/// Replace operands equal to the constant `bits` with the hoisted value.
/// Uses the same per-position immediate-encodability rule as the collector:
/// a constant that is free as an add/cmp immediate keeps its immediate form
/// there even when it was force-hoisted for a mul operand elsewhere.
fn rewrite_int_operands(inst: &mut Instruction, bits: u64, new_val: u32) {
    match inst {
        Instruction::BinOp { op, lhs, rhs, .. } => {
            let needs_reg = matches!(
                op,
                IrBinOp::Mul | IrBinOp::SDiv | IrBinOp::UDiv | IrBinOp::SRem | IrBinOp::URem
            );
            for opnd in [lhs, rhs] {
                if large_int_const(opnd, needs_reg) == Some(bits) {
                    *opnd = Operand::Value(Value(new_val));
                }
            }
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            for opnd in [lhs, rhs] {
                if large_int_const(opnd, false) == Some(bits) {
                    *opnd = Operand::Value(Value(new_val));
                }
            }
        }
        _ => {}
    }
}
