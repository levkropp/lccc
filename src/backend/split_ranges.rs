//! Live range splitting for call-spanning values.
//!
//! Splits high-priority call-spanning values by inserting explicit
//! Store/Load instructions around Call instructions. This converts
//! one long call-spanning interval into multiple short non-call-spanning
//! intervals that can use caller-saved registers (Phase 2 allocation).

use crate::common::fx_hash::FxHashSet;
use crate::common::types::{IrType, AddressSpace};
use crate::ir::reexports::{IrFunction, Instruction, Operand, Value, Terminator};

/// Split call-spanning live ranges in a function.
/// Returns the number of values split.
pub fn split_call_spanning_ranges(func: &mut IrFunction, max_splits: usize) -> usize {
    if func.blocks.is_empty() || max_splits == 0 {
        return 0;
    }

    // Step 1: Compute liveness to find call-spanning values and their ranges
    let liveness = super::liveness::compute_live_intervals(func);
    if liveness.call_points.is_empty() {
        return 0;
    }

    // Build alloca set to exclude
    let alloca_set: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|i| match i {
            Instruction::Alloca { dest, .. } => Some(dest.0),
            _ => None,
        })
        .collect();

    // Step 2: Find eligible values with cost/benefit analysis
    let mut candidates: Vec<SplitCandidate> = Vec::new();

    for iv in &liveness.intervals {
        if alloca_set.contains(&iv.value_id) { continue; }
        if iv.end <= iv.start { continue; }

        // Count calls within the value's live range
        let start_idx = liveness.call_points.partition_point(|&cp| cp < iv.start);
        let mut calls_in_range = 0u32;
        let mut idx = start_idx;
        while idx < liveness.call_points.len() && liveness.call_points[idx] <= iv.end {
            calls_in_range += 1;
            idx += 1;
        }
        if calls_in_range == 0 { continue; }

        // Count uses
        let vid = iv.value_id;
        let mut use_count = 0u32;
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst_uses_value(inst, vid) { use_count += 1; }
            }
        }

        // Cost/benefit: each call within range costs 2 instructions (Store+Load).
        // Each use that gets a register saves ~2 instructions (vs spill/reload path).
        // Only split if expected savings > cost.
        // Savings = use_count * 2 (optimistic: assumes all uses benefit from register)
        // Cost = calls_in_range * 2 (Store+Load pairs)
        if use_count * 2 <= calls_in_range * 2 + 4 {
            continue; // Not enough benefit
        }

        // Only split values whose ALL uses are in a single block.
        // Cross-block splits require dominance analysis and phi insertion.
        let mut def_block: Option<usize> = None;
        let mut all_same_block = true;
        for (bi, block) in func.blocks.iter().enumerate() {
            let defined_here = block.instructions.iter().any(|i| {
                i.dest().map_or(false, |d| d.0 == vid)
            });
            let used_here = block.instructions.iter().any(|i| inst_uses_value(i, vid));
            let used_in_term = match &block.terminator {
                Terminator::Return(Some(Operand::Value(v))) => v.0 == vid,
                Terminator::CondBranch { cond: Operand::Value(v), .. } => v.0 == vid,
                Terminator::Switch { val: Operand::Value(v), .. } => v.0 == vid,
                _ => false,
            };
            if defined_here {
                def_block = Some(bi);
            }
            if used_here || used_in_term {
                if let Some(db) = def_block {
                    if bi != db { all_same_block = false; break; }
                } else {
                    // Used before defined → cross-block (e.g., phi incoming)
                    all_same_block = false; break;
                }
            }
        }
        if !all_same_block { continue; }

        candidates.push(SplitCandidate {
            value_id: vid,
            use_count,
            calls_in_range,
            start: iv.start,
            end: iv.end,
        });
    }

    // Sort by net benefit (uses - calls) descending
    candidates.sort_by(|a, b| {
        let a_benefit = a.use_count as i64 - a.calls_in_range as i64;
        let b_benefit = b.use_count as i64 - b.calls_in_range as i64;
        b_benefit.cmp(&a_benefit)
    });

    // Step 3: Map instruction positions to (block_idx, inst_idx) for call lookup
    let mut call_positions: Vec<(u32, usize, usize)> = Vec::new(); // (program_point, block_idx, inst_idx)
    {
        let mut point = 0u32;
        for (bi, block) in func.blocks.iter().enumerate() {
            for (ii, inst) in block.instructions.iter().enumerate() {
                if matches!(inst, Instruction::Call { .. } | Instruction::CallIndirect { .. }) {
                    call_positions.push((point, bi, ii));
                }
                point += 1;
            }
            point += 1; // terminator
        }
    }

    // Step 4: Split top-N candidates
    let mut splits = 0;
    let mut next_val = func.next_value_id;

    for cand in candidates.iter().take(max_splits) {
        let val_type = match find_value_type(func, cand.value_id) {
            Some(t) => t,
            None => continue,
        };
        if val_type.is_float() || val_type.is_long_double() || val_type.is_128bit() {
            continue;
        }

        // Find calls within this value's range
        let calls_in_range: Vec<(usize, usize)> = call_positions.iter()
            .filter(|(pt, _, _)| *pt >= cand.start && *pt <= cand.end)
            .map(|(_, bi, ii)| (*bi, *ii))
            .collect();

        if calls_in_range.is_empty() { continue; }

        // Create spill alloca in entry block
        let alloca_val = Value(next_val);
        next_val += 1;
        func.blocks[0].instructions.insert(0, Instruction::Alloca {
            dest: alloca_val,
            ty: val_type,
            size: val_type.size(),
            align: 0,
            volatile: false,
        });
        if !func.blocks[0].source_spans.is_empty() {
            func.blocks[0].source_spans.insert(0, crate::common::source::Span::dummy());
        }
        // Adjust call positions for blocks[0] since we inserted at index 0
        // (all instruction indices in block 0 shift by 1)

        // Group calls by block
        let mut block_calls: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for &(bi, ii) in &calls_in_range {
            // Adjust for the alloca we just inserted in block 0
            let adjusted_ii = if bi == 0 { ii + 1 } else { ii };
            block_calls.entry(bi).or_default().push(adjusted_ii);
        }

        let mut any_split = false;
        for (&block_idx, call_indices) in &block_calls {
            // Check value is actually used in this block
            let used = func.blocks[block_idx].instructions.iter()
                .any(|inst| inst_uses_value(inst, cand.value_id));
            if !used { continue; }

            // Rebuild the block's instruction list with Store/Load around calls
            let old_insts = std::mem::take(&mut func.blocks[block_idx].instructions);
            let call_set: FxHashSet<usize> = call_indices.iter().copied().collect();
            let mut new_insts = Vec::with_capacity(old_insts.len() + call_indices.len() * 2);
            let mut current_val = cand.value_id;

            for (ii, inst) in old_insts.into_iter().enumerate() {
                if call_set.contains(&ii) {
                    // Store before call
                    new_insts.push(Instruction::Store {
                        val: Operand::Value(Value(current_val)),
                        ptr: alloca_val,
                        ty: val_type,
                        seg_override: AddressSpace::Default,
                    });
                    // The call
                    new_insts.push(inst);
                    // Load after call
                    let new_val = Value(next_val);
                    next_val += 1;
                    new_insts.push(Instruction::Load {
                        dest: new_val,
                        ptr: alloca_val,
                        ty: val_type,
                        seg_override: AddressSpace::Default,
                    });
                    current_val = new_val.0;
                    any_split = true;
                } else {
                    // Replace uses of original value with current_val (same-block only)
                    if current_val != cand.value_id {
                        let mut inst = inst;
                        replace_value_uses(&mut inst, cand.value_id, Value(current_val));
                        new_insts.push(inst);
                    } else {
                        new_insts.push(inst);
                    }
                }
            }

            // Replace uses in the terminator (same-block)
            if current_val != cand.value_id {
                replace_terminator_uses(
                    &mut func.blocks[block_idx].terminator,
                    cand.value_id,
                    Value(current_val),
                );
            }

            func.blocks[block_idx].instructions = new_insts;
            func.blocks[block_idx].source_spans.clear();
        }

        if any_split { splits += 1; }
    }

    func.next_value_id = next_val;
    splits
}

struct SplitCandidate {
    value_id: u32,
    use_count: u32,
    calls_in_range: u32,
    start: u32,
    end: u32,
}

fn find_value_type(func: &IrFunction, val_id: u32) -> Option<IrType> {
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::BinOp { dest, ty, .. } |
                Instruction::UnaryOp { dest, ty, .. } |
                Instruction::Load { dest, ty, .. } |
                Instruction::Cmp { dest, ty, .. } => {
                    if dest.0 == val_id { return Some(*ty); }
                }
                Instruction::Cast { dest, to_ty, .. } => {
                    if dest.0 == val_id { return Some(*to_ty); }
                }
                Instruction::Copy { dest, .. } |
                Instruction::GetElementPtr { dest, .. } |
                Instruction::GlobalAddr { dest, .. } |
                Instruction::LabelAddr { dest, .. } => {
                    if dest.0 == val_id { return Some(IrType::Ptr); }
                }
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    if let Some(d) = info.dest {
                        if d.0 == val_id { return Some(info.return_type); }
                    }
                }
                Instruction::Select { dest, ty, .. } |
                Instruction::Phi { dest, ty, .. } => {
                    if dest.0 == val_id { return Some(*ty); }
                }
                _ => {}
            }
        }
    }
    None
}

fn inst_uses_value(inst: &Instruction, val_id: u32) -> bool {
    let check = |op: &Operand| matches!(op, Operand::Value(v) if v.0 == val_id);
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => check(lhs) || check(rhs),
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } | Instruction::Copy { src, .. } => check(src),
        Instruction::Store { val, ptr, .. } => check(val) || ptr.0 == val_id,
        Instruction::Load { ptr, .. } => ptr.0 == val_id,
        Instruction::GetElementPtr { base, offset, .. } => base.0 == val_id || check(offset),
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => info.args.iter().any(|a| check(a)),
        Instruction::Select { cond, true_val, false_val, .. } => check(cond) || check(true_val) || check(false_val),
        _ => false,
    }
}

fn replace_value_uses(inst: &mut Instruction, old_id: u32, new_val: Value) {
    let replace = |op: &mut Operand| {
        if let Operand::Value(v) = op { if v.0 == old_id { *v = new_val; } }
    };
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => { replace(lhs); replace(rhs); }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } | Instruction::Copy { src, .. } => { replace(src); }
        Instruction::Store { val, ptr, .. } => { replace(val); if ptr.0 == old_id { *ptr = new_val; } }
        Instruction::Load { ptr, .. } => { if ptr.0 == old_id { *ptr = new_val; } }
        Instruction::GetElementPtr { base, offset, .. } => { if base.0 == old_id { *base = new_val; } replace(offset); }
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => { for a in &mut info.args { replace(a); } }
        Instruction::Select { cond, true_val, false_val, .. } => { replace(cond); replace(true_val); replace(false_val); }
        Instruction::Phi { incoming, .. } => { for (op, _) in incoming { replace(op); } }
        _ => {}
    }
}

fn replace_terminator_uses(term: &mut Terminator, old_id: u32, new_val: Value) {
    let replace = |op: &mut Operand| {
        if let Operand::Value(v) = op { if v.0 == old_id { *v = new_val; } }
    };
    match term {
        Terminator::Return(Some(op)) => replace(op),
        Terminator::CondBranch { cond, .. } => replace(cond),
        Terminator::Switch { val, .. } => replace(val),
        _ => {}
    }
}
