//! Live range splitting for call-spanning values.
//!
//! Demotes high-priority call-spanning values to stack allocas, then lets
//! mem2reg re-promote them with proper phi insertion. This creates short-lived
//! SSA values between calls that can use caller-saved registers.

use crate::common::fx_hash::FxHashSet;
use crate::common::types::{IrType, AddressSpace};
use crate::ir::reexports::{IrFunction, Instruction, Operand, Value, Terminator};

/// Split call-spanning live ranges by demoting to alloca + mem2reg.
/// Returns the number of values split.
pub fn split_call_spanning_ranges(func: &mut IrFunction, max_splits: usize) -> usize {
    if func.blocks.is_empty() || max_splits == 0 {
        return 0;
    }

    // Step 1: Compute liveness
    let liveness = super::liveness::compute_live_intervals(func);
    if liveness.call_points.is_empty() {
        return 0;
    }

    let alloca_set: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|i| match i {
            Instruction::Alloca { dest, .. } => Some(dest.0),
            _ => None,
        })
        .collect();

    // Step 2: Find eligible call-spanning values with cost/benefit
    let mut candidates: Vec<(u32, u32, u32)> = Vec::new(); // (value_id, uses, calls_in_range)

    for iv in &liveness.intervals {
        if alloca_set.contains(&iv.value_id) { continue; }
        if iv.end <= iv.start { continue; }

        // Count calls in range
        let start_idx = liveness.call_points.partition_point(|&cp| cp < iv.start);
        let mut calls = 0u32;
        let mut idx = start_idx;
        while idx < liveness.call_points.len() && liveness.call_points[idx] <= iv.end {
            calls += 1;
            idx += 1;
        }
        if calls == 0 { continue; }

        // Count uses
        let vid = iv.value_id;
        let mut uses = 0u32;
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst_uses_value(inst, vid) { uses += 1; }
            }
        }

        // Skip phi-defined values
        let is_phi = func.blocks.iter().any(|b| {
            b.instructions.iter().any(|i| matches!(i, Instruction::Phi { dest, .. } if dest.0 == vid))
        });
        if is_phi { continue; }

        // Cost/benefit: demoting adds 1 Store (at def) + 1 Load per use.
        // Benefit: each Load creates a short-lived value eligible for caller-saved.
        // Only split if the value has enough uses to justify the overhead.
        // Each use with a caller-saved register saves ~2 instructions vs spill path.
        // The Store at def costs 1 instruction. Each Load costs 1.
        // Break-even: uses * 1 (load overhead) < uses * 2 (register savings) → always true
        // But we also need the value to actually GET a register. With 6 caller-saved
        // regs and many short intervals competing, some won't get registers.
        // Conservatively require uses > calls + 4.
        if uses <= calls + 4 { continue; }

        candidates.push((vid, uses, calls));
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by use count desc

    // Step 3: Demote top-N values to allocas
    let mut splits = 0;
    let mut next_val = func.next_value_id;

    for &(val_id, _uses, _calls) in candidates.iter().take(max_splits) {
        let val_type = match find_value_type(func, val_id) {
            Some(t) => t,
            None => continue,
        };
        if val_type.is_float() || val_type.is_long_double() || val_type.is_128bit() {
            continue;
        }

        // Create alloca in entry block
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

        // Find the definition and insert Store right after it
        let mut def_found = false;
        for block in &mut func.blocks {
            let def_pos = block.instructions.iter().position(|i| {
                i.dest().map_or(false, |d| d.0 == val_id)
            });
            if let Some(pos) = def_pos {
                // Insert Store(val, alloca) right after the definition
                block.instructions.insert(pos + 1, Instruction::Store {
                    val: Operand::Value(Value(val_id)),
                    ptr: alloca_val,
                    ty: val_type,
                    seg_override: AddressSpace::Default,
                });
                if !block.source_spans.is_empty() && pos + 1 < block.source_spans.len() {
                    block.source_spans.insert(pos + 1, crate::common::source::Span::dummy());
                }
                def_found = true;
                break;
            }
        }
        if !def_found { continue; }

        // Replace all USES of val_id with Load(alloca)
        // (except the Store we just inserted, and except the definition itself)
        for block in &mut func.blocks {
            let mut i = 0;
            let mut insertions: Vec<(usize, Value)> = Vec::new();

            while i < block.instructions.len() {
                let inst = &block.instructions[i];

                // Skip the definition itself
                if inst.dest().map_or(false, |d| d.0 == val_id) {
                    i += 1;
                    // Skip the Store we inserted right after
                    if i < block.instructions.len() {
                        if let Instruction::Store { ptr, .. } = &block.instructions[i] {
                            if ptr.0 == alloca_val.0 { i += 1; }
                        }
                    }
                    continue;
                }

                // Skip Store to our alloca
                if let Instruction::Store { ptr, .. } = inst {
                    if ptr.0 == alloca_val.0 { i += 1; continue; }
                }

                // Check if this instruction uses val_id
                if inst_uses_value(inst, val_id) {
                    // Create a Load and replace the use
                    let new_val = Value(next_val);
                    next_val += 1;
                    insertions.push((i, new_val));
                }

                i += 1;
            }

            // Insert Loads BEFORE each use (in reverse order to preserve indices)
            for &(pos, new_val) in insertions.iter().rev() {
                block.instructions.insert(pos, Instruction::Load {
                    dest: new_val,
                    ptr: alloca_val,
                    ty: val_type,
                    seg_override: AddressSpace::Default,
                });
                // Replace uses in the instruction that follows
                let use_inst = &mut block.instructions[pos + 1];
                replace_value_uses(use_inst, val_id, new_val);
            }

            // Handle terminator uses
            if term_uses_value(&block.terminator, val_id) {
                let new_val = Value(next_val);
                next_val += 1;
                block.instructions.push(Instruction::Load {
                    dest: new_val,
                    ptr: alloca_val,
                    ty: val_type,
                    seg_override: AddressSpace::Default,
                });
                replace_terminator_uses(&mut block.terminator, val_id, new_val);
            }

            block.source_spans.clear();
        }

        splits += 1;
    }

    func.next_value_id = next_val;

    // Step 4: Run mem2reg to re-promote the new allocas to SSA form
    // This inserts phi nodes at merge points and optimizes away redundant loads
    if splits > 0 {
        crate::ir::mem2reg::promote::promote_function(func, false);
    }

    splits
}

fn find_value_type(func: &IrFunction, val_id: u32) -> Option<IrType> {
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::BinOp { dest, ty, .. } | Instruction::UnaryOp { dest, ty, .. }
                | Instruction::Load { dest, ty, .. } | Instruction::Cmp { dest, ty, .. } => {
                    if dest.0 == val_id { return Some(*ty); }
                }
                Instruction::Cast { dest, to_ty, .. } => {
                    if dest.0 == val_id { return Some(*to_ty); }
                }
                Instruction::Copy { dest, .. } | Instruction::GetElementPtr { dest, .. }
                | Instruction::GlobalAddr { dest, .. } | Instruction::LabelAddr { dest, .. } => {
                    if dest.0 == val_id { return Some(IrType::Ptr); }
                }
                Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
                    if let Some(d) = info.dest { if d.0 == val_id { return Some(info.return_type); } }
                }
                Instruction::Select { dest, ty, .. } | Instruction::Phi { dest, ty, .. } => {
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

fn term_uses_value(term: &Terminator, val_id: u32) -> bool {
    match term {
        Terminator::Return(Some(Operand::Value(v))) => v.0 == val_id,
        Terminator::CondBranch { cond: Operand::Value(v), .. } => v.0 == val_id,
        Terminator::Switch { val: Operand::Value(v), .. } => v.0 == val_id,
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
