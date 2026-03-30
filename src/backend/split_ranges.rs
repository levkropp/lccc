//! Live range splitting for call-spanning values.
//!
//! For high-priority call-spanning values, inserts Store/Load pairs around
//! Call instructions. After each call, a new SSA value is created via Load.
//! Uses after the call (same-block only) are replaced with the new value.
//! mem2reg then promotes the spill alloca to insert phi nodes at merge points.

use crate::common::fx_hash::FxHashSet;
use crate::common::types::{IrType, AddressSpace};
use crate::ir::reexports::{IrFunction, Instruction, Operand, Value, Terminator};

pub fn split_call_spanning_ranges(func: &mut IrFunction, max_splits: usize) -> usize {
    if func.blocks.is_empty() || max_splits == 0 { return 0; }

    let liveness = super::liveness::compute_live_intervals(func);
    if liveness.call_points.is_empty() { return 0; }

    let alloca_set: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|i| match i { Instruction::Alloca { dest, .. } => Some(dest.0), _ => None })
        .collect();

    // Find candidates
    let mut candidates: Vec<(u32, u32, u32, u32, u32)> = Vec::new(); // (vid, uses, calls, start, end)
    for iv in &liveness.intervals {
        if alloca_set.contains(&iv.value_id) || iv.end <= iv.start { continue; }
        let vid = iv.value_id;

        let start_idx = liveness.call_points.partition_point(|&cp| cp < iv.start);
        let mut calls = 0u32;
        let mut idx = start_idx;
        while idx < liveness.call_points.len() && liveness.call_points[idx] <= iv.end { calls += 1; idx += 1; }
        if calls == 0 { continue; }

        let is_phi = func.blocks.iter().any(|b| b.instructions.iter().any(|i| matches!(i, Instruction::Phi { dest, .. } if dest.0 == vid)));
        if is_phi { continue; }

        // Only split values used entirely within a single block (safe)
        let def_block = func.blocks.iter().position(|b| {
            b.instructions.iter().any(|i| i.dest().map_or(false, |d| d.0 == vid))
        });
        let all_same = if let Some(db) = def_block {
            func.blocks.iter().enumerate().all(|(bi, b)| {
                bi == db || (!b.instructions.iter().any(|i| inst_uses_value(i, vid))
                    && !match &b.terminator {
                        Terminator::Return(Some(Operand::Value(v))) => v.0 == vid,
                        Terminator::CondBranch { cond: Operand::Value(v), .. } => v.0 == vid,
                        _ => false,
                    })
            })
        } else { false };
        if !all_same { continue; }

        let mut uses = 0u32;
        for b in &func.blocks { for i in &b.instructions { if inst_uses_value(i, vid) { uses += 1; } } }
        if uses <= calls + 2 { continue; }

        candidates.push((vid, uses, calls, iv.start, iv.end));
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1));


    let mut splits = 0;
    let mut next_val = func.next_value_id;

    for &(val_id, _uses, _calls, range_start, range_end) in candidates.iter().take(max_splits) {
        let val_type = match find_value_type(func, val_id) {
            Some(t) if !t.is_float() && !t.is_long_double() && !t.is_128bit() => t,
            _ => continue,
        };

        // Create spill alloca
        let alloca_val = Value(next_val); next_val += 1;
        func.blocks[0].instructions.insert(0, Instruction::Alloca {
            dest: alloca_val, ty: val_type, size: val_type.size(), align: 0, volatile: false,
        });
        func.blocks[0].source_spans.clear();

        // Also store the initial value right after its definition
        let def_block = func.blocks.iter().position(|b| {
            b.instructions.iter().any(|i| i.dest().map_or(false, |d| d.0 == val_id))
        });
        if let Some(dbi) = def_block {
            let def_pos = func.blocks[dbi].instructions.iter().position(|i| {
                i.dest().map_or(false, |d| d.0 == val_id)
            });
            if let Some(dp) = def_pos {
                func.blocks[dbi].instructions.insert(dp + 1, Instruction::Store {
                    val: Operand::Value(Value(val_id)), ptr: alloca_val,
                    ty: val_type, seg_override: AddressSpace::Default,
                });
                func.blocks[dbi].source_spans.clear();
            }
        }

        // For each block, find calls and process
        let mut any_split = false;
        for bi in 0..func.blocks.len() {
            // Check if block has any calls AND uses of our value
            let has_call = func.blocks[bi].instructions.iter().any(|i|
                matches!(i, Instruction::Call { .. } | Instruction::CallIndirect { .. }));
            if !has_call { continue; }
            let has_use = func.blocks[bi].instructions.iter().any(|i| inst_uses_value(i, val_id));
            if !has_use { continue; }

            // Find call instruction indices in this block (by scanning, not precomputed)
            let call_set: FxHashSet<usize> = func.blocks[bi].instructions.iter().enumerate()
                .filter(|(_, i)| matches!(i, Instruction::Call { .. } | Instruction::CallIndirect { .. }))
                .map(|(idx, _)| idx)
                .collect();
            let old_insts = std::mem::take(&mut func.blocks[bi].instructions);
            let mut new_insts = Vec::with_capacity(old_insts.len() + call_set.len() * 2);
            let mut current_val = val_id;

            for (ii, inst) in old_insts.into_iter().enumerate() {
                if call_set.contains(&ii) {
                    // Store current value before call
                    if current_val == val_id {
                        // Original value — already stored at def, but re-store in case
                        // it was modified (it shouldn't be for SSA, but be safe)
                        new_insts.push(Instruction::Store {
                            val: Operand::Value(Value(current_val)), ptr: alloca_val,
                            ty: val_type, seg_override: AddressSpace::Default,
                        });
                    }
                    // The call
                    new_insts.push(inst);
                    // Load after call
                    let new_val = Value(next_val); next_val += 1;
                    new_insts.push(Instruction::Load {
                        dest: new_val, ptr: alloca_val,
                        ty: val_type, seg_override: AddressSpace::Default,
                    });
                    current_val = new_val.0;
                    any_split = true;
                } else if current_val != val_id {
                    let mut inst = inst;
                    replace_value_uses(&mut inst, val_id, Value(current_val));
                    new_insts.push(inst);
                } else {
                    new_insts.push(inst);
                }
            }

            if current_val != val_id {
                replace_terminator_uses(&mut func.blocks[bi].terminator, val_id, Value(current_val));

                // Update phi incoming values in successor blocks that reference
                // val_id from this block — they should use current_val instead.
                let this_label = func.blocks[bi].label;
                let succs: Vec<usize> = match &func.blocks[bi].terminator {
                    Terminator::Branch(lbl) => {
                        func.blocks.iter().position(|b| b.label == *lbl).into_iter().collect()
                    }
                    Terminator::CondBranch { true_label, false_label, .. } => {
                        let mut v = Vec::new();
                        if let Some(p) = func.blocks.iter().position(|b| b.label == *true_label) { v.push(p); }
                        if let Some(p) = func.blocks.iter().position(|b| b.label == *false_label) { if !v.contains(&p) { v.push(p); } }
                        v
                    }
                    _ => Vec::new(),
                };

                for succ_idx in succs {
                    for inst in &mut func.blocks[succ_idx].instructions {
                        if let Instruction::Phi { incoming, .. } = inst {
                            for (op, from_label) in incoming {
                                if *from_label == this_label {
                                    if let Operand::Value(v) = op {
                                        if v.0 == val_id { *v = Value(current_val); }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            func.blocks[bi].instructions = new_insts;
            func.blocks[bi].source_spans.clear();
        }

        if any_split { splits += 1; }
    }

    func.next_value_id = next_val;

    // Note: mem2reg promotion of spill allocas is disabled because it
    // interacts incorrectly with the splitting pattern. The allocas
    // stay as stack slots — they're still useful because the Load
    // after each call creates a short-lived value that can use
    // caller-saved registers (Phase 2 allocation).

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

fn replace_value_uses(inst: &mut Instruction, old: u32, new: Value) {
    let r = |op: &mut Operand| { if let Operand::Value(v) = op { if v.0 == old { *v = new; } } };
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => { r(lhs); r(rhs); }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } | Instruction::Copy { src, .. } => r(src),
        Instruction::Store { val, ptr, .. } => { r(val); if ptr.0 == old { *ptr = new; } }
        Instruction::Load { ptr, .. } => { if ptr.0 == old { *ptr = new; } }
        Instruction::GetElementPtr { base, offset, .. } => { if base.0 == old { *base = new; } r(offset); }
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => { for a in &mut info.args { r(a); } }
        Instruction::Select { cond, true_val, false_val, .. } => { r(cond); r(true_val); r(false_val); }
        Instruction::Phi { incoming, .. } => { for (op, _) in incoming { r(op); } }
        _ => {}
    }
}

fn replace_terminator_uses(t: &mut Terminator, old: u32, new: Value) {
    let r = |op: &mut Operand| { if let Operand::Value(v) = op { if v.0 == old { *v = new; } } };
    match t {
        Terminator::Return(Some(op)) => r(op),
        Terminator::CondBranch { cond, .. } => r(cond),
        Terminator::Switch { val, .. } => r(val),
        _ => {}
    }
}
