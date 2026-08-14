//! FP constant hoisting.
//!
//! FP constants used inside loop bodies are otherwise materialized per
//! iteration via a constant-pool literal load (`ldr d0, .LCFP_N`), which sits
//! in the loop's dependency path.  This pass materializes each distinct
//! non-zero F32/F64 constant once in the loop preheader as a `Copy`, making it
//! a register-allocatable SSA value that can stay in an FP register across the
//! whole loop.
//!
//! Example: `if (zr*zr + zi*zi > 4.0)` in a loop becomes
//!   preheader:  %c4 = copy 4.0
//!   loop:       ... cmp %x, %c4        (no per-iteration constant load)

use crate::common::fx_hash::FxHashSet;
use crate::ir::analysis;
use crate::ir::reexports::{Instruction, IrConst, IrFunction, Operand, Value};
use super::loop_analysis;

/// Run FP-constant hoisting on a function. Returns the number of constants hoisted.
pub(crate) fn run(func: &mut IrFunction) -> usize {
    if func.blocks.len() < 2 {
        return 0;
    }
    let cfg = analysis::CfgAnalysis::build(func);
    let mut loops = loop_analysis::find_natural_loops(cfg.num_blocks, &cfg.preds, &cfg.succs, &cfg.idom);
    if loops.is_empty() {
        return 0;
    }
    // Outermost first so constants used by nested loops hoist as far out as
    // possible.
    loops.sort_by_key(|l| l.body.len());
    loops.reverse();

    let mut hoisted = 0;
    for lp in &loops {
        let Some(preheader) = loop_analysis::find_preheader(lp.header, &lp.body, &cfg.preds) else {
            continue;
        };
        hoisted += hoist_in_loop(func, lp, preheader);
    }
    hoisted
}

/// A distinct FP constant to hoist, keyed by type and bit pattern.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum FpConst {
    F32(u32),
    F64(u64),
}

impl FpConst {
    fn is_zero(&self) -> bool {
        match self {
            FpConst::F32(b) => *b == 0,
            FpConst::F64(b) => *b == 0,
        }
    }
    fn to_ir_const(&self) -> IrConst {
        match self {
            FpConst::F32(b) => IrConst::F32(f32::from_bits(*b)),
            FpConst::F64(b) => IrConst::F64(f64::from_bits(*b)),
        }
    }
}

fn const_of(op: &Operand) -> Option<FpConst> {
    match op {
        Operand::Const(IrConst::F32(v)) => Some(FpConst::F32(v.to_bits())),
        Operand::Const(IrConst::F64(v)) => Some(FpConst::F64(v.to_bits())),
        _ => None,
    }
}

/// Visit operands that can hold an FP constant (the arithmetic/comparison
/// forms that appear in FP loop bodies).
fn for_each_fp_operand(inst: &Instruction, f: &mut dyn FnMut(&Operand)) {
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } => f(src),
        Instruction::Select { cond, true_val, false_val, .. } => {
            f(cond);
            f(true_val);
            f(false_val);
        }
        _ => {}
    }
}

fn hoist_in_loop(func: &mut IrFunction, lp: &loop_analysis::NaturalLoop, preheader: usize) -> usize {
    // Collect distinct non-zero FP constants used as operands in the loop body.
    let mut consts: Vec<FpConst> = Vec::new();
    let mut seen: FxHashSet<FpConst> = FxHashSet::default();
    for &bi in &lp.body {
        for inst in &func.blocks[bi].instructions {
            for_each_fp_operand(inst, &mut |op| {
                if let Some(c) = const_of(op) {
                    // Zero is already cheap (`fmov d, xzr`); no load to hoist.
                    if !c.is_zero() && seen.insert(c) {
                        consts.push(c);
                    }
                }
            });
        }
    }
    if consts.is_empty() {
        return 0;
    }

    // For each distinct constant, create a Copy in the preheader and rewrite
    // in-loop uses to reference the new value.
    let mut count = 0;
    for c in consts {
        let new_val = func.next_value_id;
        func.next_value_id += 1;
        func.blocks[preheader].instructions.push(Instruction::Copy {
            dest: Value(new_val),
            src: Operand::Const(c.to_ir_const()),
        });
        count += 1;

        // Rewrite uses within the loop body.
        for &bi in &lp.body {
            for inst in &mut func.blocks[bi].instructions {
                rewrite_fp_operands(inst, c, new_val);
            }
        }
    }
    count
}

/// Replace operands matching `c` with the hoisted value `new_val`.
fn rewrite_fp_operands(inst: &mut Instruction, c: FpConst, new_val: u32) {
    let sub = |op: &mut Operand| {
        if const_of(op) == Some(c) {
            *op = Operand::Value(Value(new_val));
        }
    };
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => {
            sub(lhs);
            sub(rhs);
        }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } => sub(src),
        Instruction::Select { cond, true_val, false_val, .. } => {
            sub(cond);
            sub(true_val);
            sub(false_val);
        }
        _ => {}
    }
}
