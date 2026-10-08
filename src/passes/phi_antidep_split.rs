//! Split loop-phi anti-dependencies so backedge coalescing can kill the
//! in-chain register copy.
//!
//! In a loop recurrence like
//!   p    = phi(init, newv)      // loop header
//!   newv = fadd(p_sq, cr)       // next value, computed mid-body
//!   use  = fmul(p, two)         // OLD p still needed after newv is born
//! the backend cannot coalesce p and newv into one register: the fmul reads
//! p after newv's definition, so sharing would clobber it. The phi-coalesce
//! detector blocks the pair ("phi dest used after src def") and the backedge
//! copy survives — inside the recurrence's serial dependency chain.
//!
//! Rewriting the post-def uses of p to a copy made just before newv's def is
//! always sound (the copy reads p while it still holds the current value)
//! and unblocks coalescing; the copy itself feeds only off-chain consumers.
//! GCC -O2 emits exactly this shape for mandelbrot: an off-chain fmov at the
//! top of the loop with the accumulator updated in place (-14% there).

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::analysis::CfgAnalysis;
use crate::ir::reexports::{BlockId, Instruction, IrFunction, Operand, Terminator, Value};
use super::loop_analysis;

/// Replace uses of `old` with `new` in an instruction's operand positions
/// (never the destination). Returns true if anything changed.
fn replace_use_in_inst(inst: &mut Instruction, old: u32, new: Value) -> bool {
    let mut hit = false;
    let mut op = |o: &mut Operand| -> bool {
        if matches!(o, Operand::Value(v) if v.0 == old) {
            *o = Operand::Value(new);
            true
        } else {
            false
        }
    };
    let mut val = |v: &mut Value| -> bool {
        if v.0 == old {
            *v = new;
            true
        } else {
            false
        }
    };
    match inst {
        Instruction::BinOp { lhs, rhs, .. } => {
            hit |= op(lhs);
            hit |= op(rhs);
        }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } => hit |= op(src),
        Instruction::Copy { src, .. } => hit |= op(src),
        Instruction::Cmp { lhs, rhs, .. } => {
            hit |= op(lhs);
            hit |= op(rhs);
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            hit |= op(cond);
            hit |= op(true_val);
            hit |= op(false_val);
        }
        Instruction::Store { val: v, ptr, .. } => {
            hit |= op(v);
            hit |= val(ptr);
        }
        Instruction::Load { ptr, .. } => hit |= val(ptr),
        Instruction::GetElementPtr { base, offset, .. } => {
            hit |= val(base);
            hit |= op(offset);
        }
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            for a in &mut info.args {
                hit |= op(a);
            }
        }
        _ => {}
    }
    hit
}

fn replace_use_in_term(term: &mut Terminator, old: u32, new: Value) {
    match term {
        Terminator::CondBranch { cond, .. } => {
            if matches!(cond, Operand::Value(v) if v.0 == old) {
                *cond = Operand::Value(new);
            }
        }
        Terminator::Return(Some(o)) => {
            if matches!(o, Operand::Value(v) if v.0 == old) {
                *o = Operand::Value(new);
            }
        }
        Terminator::Switch { val, .. } => {
            if matches!(val, Operand::Value(v) if v.0 == old) {
                *val = Operand::Value(new);
            }
        }
        _ => {}
    }
}

/// Does the instruction use `v` in any operand position?
fn inst_uses(inst: &Instruction, v: u32) -> bool {
    let mut found = false;
    inst.for_each_used_value(|u| {
        if u == v {
            found = true;
        }
    });
    found
}

pub(crate) fn run(func: &mut IrFunction) -> usize {
    if std::env::var("CCC_NO_PHI_ANTIDEP").is_ok() {
        return 0;
    }
    let cfg = CfgAnalysis::build(func);
    let loops = loop_analysis::find_natural_loops(
        func.blocks.len(),
        &cfg.preds,
        &cfg.succs,
        &cfg.idom,
    );
    if loops.is_empty() {
        return 0;
    }

    // Definition sites of every value (block, instruction index). Values with
    // zero or several definitions cannot be recurrence sources.
    let mut def_site: FxHashMap<u32, (usize, usize)> = FxHashMap::default();
    let mut multi_def: FxHashSet<u32> = FxHashSet::default();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if let Some(d) = inst.dest() {
                if def_site.insert(d.0, (bi, ii)).is_some() {
                    multi_def.insert(d.0);
                }
            }
        }
    }

    let mut changed = 0;
    for lp in &loops {
        // Phis live at the top of the header block.
        let header = lp.header;
        let phis: Vec<(Value, Vec<(Operand, BlockId)>)> = func.blocks[header]
            .instructions
            .iter()
            .filter_map(|inst| match inst {
                Instruction::Phi { dest, incoming, .. } => {
                    Some((*dest, incoming.clone()))
                }
                _ => None,
            })
            .collect();
        for (phi_dest, incoming) in phis {
            for (op, _from_block) in &incoming {
                let Operand::Value(back) = op else { continue };
                if multi_def.contains(&back.0) {
                    continue; // swap-cycle temporaries: leave alone
                }
                let Some(&(db, di)) = def_site.get(&back.0) else { continue };
                if !lp.body.contains(&db) {
                    continue;
                }
                // Any use of the phi dest after the backedge value's def in
                // the defining block blocks coalescing.
                let block = &func.blocks[db];
                let has_post_def_use = block.instructions[di + 1..]
                    .iter()
                    .any(|inst| inst_uses(inst, phi_dest.0));
                if !has_post_def_use {
                    continue;
                }
                // Insert `pc = copy phi_dest` just before the def and rewrite
                // all later uses (including the terminator) to pc. Verify the
                // rewrite was complete; if any use remains (an instruction
                // kind the rewriter does not cover), discard the edit.
                let pc = Value(func.next_value_id);
                let mut insts = block.instructions.clone();
                let mut term = block.terminator.clone();
                insts.insert(
                    di,
                    Instruction::Copy { dest: pc, src: Operand::Value(phi_dest) },
                );
                // Def moved to di + 1; rewrite uses after it.
                for inst in &mut insts[di + 2..] {
                    replace_use_in_inst(inst, phi_dest.0, pc);
                }
                replace_use_in_term(&mut term, phi_dest.0, pc);
                let remaining = insts[di + 2..].iter().any(|inst| inst_uses(inst, phi_dest.0))
                    || {
                        let mut f = false;
                        match &term {
                            Terminator::CondBranch { cond, .. } => {
                                f = matches!(cond, Operand::Value(v) if v.0 == phi_dest.0)
                            }
                            Terminator::Return(Some(o)) => {
                                f = matches!(o, Operand::Value(v) if v.0 == phi_dest.0)
                            }
                            Terminator::Switch { val, .. } => {
                                f = matches!(val, Operand::Value(v) if v.0 == phi_dest.0)
                            }
                            _ => {}
                        }
                        f
                    };
                if remaining {
                    continue;
                }
                func.next_value_id += 1;
                func.blocks[db].instructions = insts;
                func.blocks[db].terminator = term;
                changed += 1;
            }
        }
    }
    changed
}
