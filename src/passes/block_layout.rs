//! Block layout: reorder function blocks into reverse post-order.
//!
//! Several passes (vectorization, inlining, switch outlining) append new
//! blocks at the end of `func.blocks`, leaving the linearized block order
//! unrelated to execution order. The backend's liveness analysis numbers
//! instructions in block order, so an out-of-order loop exit block stretches
//! live intervals across unrelated code — including calls — which defeats
//! register allocation (a vector or FP accumulator that should stay in a
//! register for the whole loop gets stack-homed instead).
//!
//! Re-layout in reverse post-order keeps each loop's blocks and its exit
//! block contiguous. Block identity is the `label` field (terminators
//! reference labels, not positions), so reordering the list is
//! semantics-preserving.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::instruction::{BasicBlock, Terminator};
use crate::ir::reexports::IrFunction;

/// Reorder `func.blocks` into reverse post-order from the entry block.
/// Unreachable blocks (if any) keep their original relative order at the end.
/// Returns 1 if the order changed, 0 otherwise.
pub(crate) fn relayout_blocks_rpo(func: &mut IrFunction) -> usize {
    if func.blocks.len() < 2 {
        return 0;
    }
    let mut pos_of_label: FxHashMap<u32, usize> = FxHashMap::default();
    for (idx, block) in func.blocks.iter().enumerate() {
        pos_of_label.insert(block.label.0, idx);
    }

    // Iterative post-order DFS from the entry block. Successors are pushed in
    // natural (true, false) order so the stack explores the last successor
    // first; after the final reversal this places loop bodies right after
    // their header and exit blocks right after the loop.
    let entry = func.blocks[0].label.0;
    let mut visited: FxHashSet<u32> = FxHashSet::default();
    let mut postorder: Vec<u32> = Vec::with_capacity(func.blocks.len());
    let mut stack: Vec<(u32, bool)> = vec![(entry, false)];
    while let Some((label, processed)) = stack.pop() {
        if processed {
            postorder.push(label);
            continue;
        }
        if !visited.insert(label) {
            continue;
        }
        stack.push((label, true));
        let mut succs: Vec<u32> = Vec::new();
        if let Some(&idx) = pos_of_label.get(&label) {
            collect_successor_labels(&func.blocks[idx].terminator, &mut succs);
            // InlineAsm goto_labels are implicit control-flow edges.
            for inst in &func.blocks[idx].instructions {
                if let crate::ir::instruction::Instruction::InlineAsm { goto_labels, .. } = inst {
                    succs.extend(goto_labels.iter().map(|(_, l)| l.0));
                }
            }
        }
        for &s in &succs {
            if !visited.contains(&s) {
                stack.push((s, false));
            }
        }
    }

    postorder.reverse();
    let mut new_order: Vec<usize> = postorder.iter().map(|l| pos_of_label[l]).collect();
    let mut placed: FxHashSet<usize> = new_order.iter().copied().collect();
    for idx in 0..func.blocks.len() {
        if placed.insert(idx) {
            new_order.push(idx);
        }
    }

    if new_order.iter().enumerate().all(|(new_pos, &old_pos)| new_pos == old_pos) {
        return 0;
    }
    let mut old: Vec<Option<BasicBlock>> = func.blocks.drain(..).map(Some).collect();
    func.blocks = new_order.into_iter().map(|i| old[i].take().unwrap()).collect();
    1
}

/// Collect the target label IDs of a terminator.
fn collect_successor_labels(term: &Terminator, out: &mut Vec<u32>) {
    match term {
        Terminator::Branch(l) => out.push(l.0),
        Terminator::CondBranch { true_label, false_label, .. } => {
            out.push(true_label.0);
            out.push(false_label.0);
        }
        Terminator::IndirectBranch { possible_targets, .. } => {
            out.extend(possible_targets.iter().map(|l| l.0));
        }
        Terminator::Switch { cases, default, .. } => {
            out.extend(cases.iter().map(|(_, l)| l.0));
            out.push(default.0);
        }
        Terminator::Return(_) | Terminator::Unreachable => {}
    }
}
