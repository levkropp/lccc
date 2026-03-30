//! Live range splitting for call-spanning values.
//!
//! Uses the SSA construction algorithm (Cytron et al. 1991) to split
//! call-spanning values at call boundaries, inserting phi nodes at
//! dominance frontiers. Each segment between calls can then use
//! caller-saved registers (Phase 2 allocation).

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::{IrType, AddressSpace};
use crate::ir::reexports::*;
use crate::ir::analysis;
use std::collections::VecDeque;

pub fn split_call_spanning_ranges(func: &mut IrFunction, max_splits: usize) -> usize {
    if func.blocks.is_empty() || max_splits == 0 { return 0; }

    let liveness = super::liveness::compute_live_intervals(func);
    if liveness.call_points.is_empty() { return 0; }

    let alloca_set: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|i| match i { Instruction::Alloca { dest, .. } => Some(dest.0), _ => None })
        .collect();

    // Find candidates: call-spanning, non-phi, non-alloca, with enough uses
    let mut candidates: Vec<(u32, u32, u32)> = Vec::new(); // (vid, uses, calls)
    for iv in &liveness.intervals {
        if alloca_set.contains(&iv.value_id) || iv.end <= iv.start { continue; }
        let vid = iv.value_id;
        let si = liveness.call_points.partition_point(|&cp| cp < iv.start);
        let mut calls = 0u32;
        let mut idx = si;
        while idx < liveness.call_points.len() && liveness.call_points[idx] <= iv.end { calls += 1; idx += 1; }
        if calls == 0 { continue; }
        let is_phi = func.blocks.iter().any(|b| b.instructions.iter()
            .any(|i| matches!(i, Instruction::Phi { dest, .. } if dest.0 == vid)));
        if is_phi { continue; }
        let mut uses = 0u32;
        for b in &func.blocks { for i in &b.instructions { if inst_uses_value(i, vid) { uses += 1; } } }
        // Need many more uses than calls for the splitting to pay off:
        // each call costs ~14 bytes (Store+Load), each registered use saves ~4 bytes
        if uses < calls * 4 + 5 { continue; }
        candidates.push((vid, uses, calls));
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    if candidates.is_empty() { return 0; }

    // Compute CFG analysis for phi insertion
    let label_map = analysis::build_label_map(func);
    let (preds, succs) = analysis::build_cfg(func, &label_map);
    let idom = analysis::compute_dominators(func.blocks.len(), &preds, &succs);
    let dom_children = analysis::build_dom_tree_children(func.blocks.len(), &idom);
    let df = analysis::compute_dominance_frontiers(func.blocks.len(), &preds, &idom);

    let mut splits = 0;
    let mut next_val = func.next_value_id;
    let mut new_alloca_ids: FxHashSet<u32> = FxHashSet::default();

    for &(val_id, _uses, _calls) in candidates.iter().take(max_splits) {
        let val_type = match find_value_type(func, val_id) {
            Some(t) if !t.is_float() && !t.is_long_double() && !t.is_128bit() => t,
            _ => continue,
        };

        // Find definition block
        let def_block = func.blocks.iter().position(|b|
            b.instructions.iter().any(|i| i.dest().map_or(false, |d| d.0 == val_id)));
        let def_block = match def_block { Some(b) => b, None => continue };

        // Find blocks containing calls where the value is used
        // (these are the "re-definition" blocks — after the call, we get a new value)
        let mut redef_blocks: FxHashSet<usize> = FxHashSet::default();
        redef_blocks.insert(def_block); // original definition
        for (bi, block) in func.blocks.iter().enumerate() {
            let has_call = block.instructions.iter().any(|i|
                matches!(i, Instruction::Call { .. } | Instruction::CallIndirect { .. }));
            let has_use = block.instructions.iter().any(|i| inst_uses_value(i, val_id));
            if has_call && has_use {
                redef_blocks.insert(bi);
            }
        }

        // Iterated dominance frontier: find phi insertion points
        let mut phi_blocks: FxHashSet<usize> = FxHashSet::default();
        let mut worklist: VecDeque<usize> = redef_blocks.iter().copied().collect();
        let mut ever_in_wl: FxHashSet<usize> = redef_blocks.clone();
        while let Some(b) = worklist.pop_front() {
            if b >= df.len() { continue; }
            for &frontier_b in &df[b] {
                if phi_blocks.insert(frontier_b) {
                    if ever_in_wl.insert(frontier_b) {
                        worklist.push_back(frontier_b);
                    }
                }
            }
        }

        // Phi insertion is handled by mem2reg when it promotes the alloca.
        // We just insert Store/Load pairs and let the alloca serve as the
        // spill slot. No manual phi insertion needed.

        // Insert Store/Load around calls in blocks that use the value
        // Also track which blocks get a "redefinition" (Load after call)
        let mut block_redefs: FxHashMap<usize, Vec<(usize, Value)>> = FxHashMap::default(); // block -> [(inst_idx_after_load, new_val)]

        for bi in 0..func.blocks.len() {
            let block = &func.blocks[bi];
            let call_indices: Vec<usize> = block.instructions.iter().enumerate()
                .filter(|(_, i)| matches!(i, Instruction::Call { .. } | Instruction::CallIndirect { .. }))
                .map(|(idx, _)| idx)
                .collect();
            if call_indices.is_empty() { continue; }
            let has_use = block.instructions.iter().any(|i| inst_uses_value(i, val_id));
            if !has_use && bi != def_block { continue; }

            // Create spill alloca (one per split value)
            // Actually, use a single alloca for each value
            // We'll create it once below
        }

        // Create alloca for spill slot — insert AFTER existing allocas in entry block
        let alloca_val = Value(next_val); next_val += 1;
        new_alloca_ids.insert(alloca_val.0);
        let insert_pos = func.blocks[0].instructions.iter()
            .position(|i| !matches!(i, Instruction::Alloca { .. }))
            .unwrap_or(func.blocks[0].instructions.len());
        func.blocks[0].instructions.insert(insert_pos, Instruction::Alloca {
            dest: alloca_val, ty: val_type, size: val_type.size(), align: 0, volatile: false,
        });
        func.blocks[0].source_spans.clear();

        // Store right after the definition
        {
            let block = &mut func.blocks[def_block];
            let def_pos = block.instructions.iter().position(|i|
                i.dest().map_or(false, |d| d.0 == val_id)).unwrap();
            block.instructions.insert(def_pos + 1, Instruction::Store {
                val: Operand::Value(Value(val_id)), ptr: alloca_val,
                ty: val_type, seg_override: AddressSpace::Default,
            });
            block.source_spans.clear();
        }

        // Only insert Store/Load in blocks that are AFTER the def block
        // in the dominator tree (where val_id is guaranteed to be defined)
        for bi in 0..func.blocks.len() {
            // Only insert in blocks dominated by the def block
            let mut dominated = bi == def_block;
            if !dominated {
                let mut b = bi;
                while b < idom.len() && idom[b] != b && idom[b] != usize::MAX {
                    if idom[b] == def_block { dominated = true; break; }
                    b = idom[b];
                }
            }
            if !dominated { continue; }

            let call_indices: Vec<usize> = func.blocks[bi].instructions.iter().enumerate()
                .filter(|(_, i)| matches!(i, Instruction::Call { .. } | Instruction::CallIndirect { .. }))
                .map(|(idx, _)| idx)
                .collect();
            if call_indices.is_empty() { continue; }

            // In the def block, only insert around calls AFTER the definition
            let def_pos_in_block = if bi == def_block {
                func.blocks[bi].instructions.iter().position(|i|
                    i.dest().map_or(false, |d| d.0 == val_id)).unwrap_or(0)
            } else {
                0 // all calls are after def for dominated blocks
            };

            // Filter to calls after the def AND where value is used after the call
            let insts = &func.blocks[bi].instructions;
            let call_indices: Vec<usize> = call_indices.into_iter()
                .filter(|&ci| ci > def_pos_in_block)
                .filter(|&ci| {
                    // Check if value is used AFTER this call in the block
                    (ci + 1..insts.len()).any(|j| inst_uses_value(&insts[j], val_id))
                })
                .collect();
            if call_indices.is_empty() { continue; }

            if std::env::var("CCC_DEBUG_SPLIT").is_ok() {
                eprintln!("[SPLIT] Block {} has {} calls after def for Value({})",
                    bi, call_indices.len(), val_id);
            }

            // Insert Store before + Load after each call
            let mut offset = 0i32;
            for &ci in &call_indices {
                let adj_ci = (ci as i32 + offset) as usize;
                func.blocks[bi].instructions.insert(adj_ci, Instruction::Store {
                    val: Operand::Value(Value(val_id)), ptr: alloca_val,
                    ty: val_type, seg_override: AddressSpace::Default,
                });
                offset += 1;
                let adj_ci = adj_ci + 1;
                let new_val = Value(next_val); next_val += 1;
                func.blocks[bi].instructions.insert(adj_ci + 1, Instruction::Load {
                    dest: new_val, ptr: alloca_val,
                    ty: val_type, seg_override: AddressSpace::Default,
                });
                offset += 1;
                block_redefs.entry(bi).or_default().push((adj_ci + 1, new_val));
            }
            func.blocks[bi].source_spans.clear();
        }

        // Now run mem2reg to promote the alloca — this handles ALL the SSA
        // reconstruction: phi insertion, use replacement, cross-block renaming
        splits += 1;
    }

    func.next_value_id = next_val;

    // Promote spill allocas via mem2reg. To avoid re-promoting existing
    // allocas (which may have non-promotable patterns), temporarily mark
    // all PRE-EXISTING non-param allocas as volatile. Our new allocas
    // are non-volatile and will be promoted by mem2reg.
    if splits > 0 && std::env::var("CCC_NO_SPLIT_MEM2REG").is_err() {
        // Mark existing allocas as volatile (skip first num_params)
        let num_params = func.params.len();
        let mut alloca_idx = 0;
        let mut made_volatile: Vec<u32> = Vec::new();
        for inst in &mut func.blocks[0].instructions {
            if let Instruction::Alloca { dest, volatile, .. } = inst {
                if alloca_idx >= num_params && !*volatile && !new_alloca_ids.contains(&dest.0) {
                    *volatile = true;
                    made_volatile.push(dest.0);
                }
                alloca_idx += 1;
            }
        }
        // Also mark allocas in non-entry blocks
        for block in &mut func.blocks[1..] {
            for inst in &mut block.instructions {
                if let Instruction::Alloca { dest, volatile, .. } = inst {
                    if !*volatile && !new_alloca_ids.contains(&dest.0) {
                        *volatile = true;
                        made_volatile.push(dest.0);
                    }
                }
            }
        }

        crate::ir::mem2reg::promote::promote_function(func, false);

        // Restore volatility
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if let Instruction::Alloca { dest, volatile, .. } = inst {
                    if made_volatile.contains(&dest.0) {
                        *volatile = false;
                    }
                }
            }
        }
    }

    splits
}

fn find_value_type(func: &IrFunction, val_id: u32) -> Option<IrType> {
    for b in &func.blocks {
        for i in &b.instructions {
            match i {
                Instruction::BinOp { dest, ty, .. } | Instruction::UnaryOp { dest, ty, .. }
                | Instruction::Load { dest, ty, .. } | Instruction::Cmp { dest, ty, .. } =>
                    { if dest.0 == val_id { return Some(*ty); } }
                Instruction::Cast { dest, to_ty, .. } =>
                    { if dest.0 == val_id { return Some(*to_ty); } }
                Instruction::Copy { dest, .. } | Instruction::GetElementPtr { dest, .. }
                | Instruction::GlobalAddr { dest, .. } =>
                    { if dest.0 == val_id { return Some(IrType::Ptr); } }
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } =>
                    { if info.dest.map_or(false, |d| d.0 == val_id) { return Some(info.return_type); } }
                Instruction::Select { dest, ty, .. } | Instruction::Phi { dest, ty, .. } =>
                    { if dest.0 == val_id { return Some(*ty); } }
                _ => {}
            }
        }
    }
    None
}

fn inst_uses_value(inst: &Instruction, v: u32) -> bool {
    let c = |op: &Operand| matches!(op, Operand::Value(val) if val.0 == v);
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => c(lhs) || c(rhs),
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } | Instruction::Copy { src, .. } => c(src),
        Instruction::Store { val, ptr, .. } => c(val) || ptr.0 == v,
        Instruction::Load { ptr, .. } => ptr.0 == v,
        Instruction::GetElementPtr { base, offset, .. } => base.0 == v || c(offset),
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => info.args.iter().any(|a| c(a)),
        Instruction::Select { cond, true_val, false_val, .. } => c(cond) || c(true_val) || c(false_val),
        _ => false,
    }
}
