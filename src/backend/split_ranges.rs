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

    // Step 1: Compute liveness to find call-spanning values
    let liveness = super::liveness::compute_live_intervals(func);
    if liveness.call_points.is_empty() {
        return 0; // No calls → nothing to split
    }

    // Step 2: Find eligible call-spanning values sorted by use count
    let alloca_set: FxHashSet<u32> = func.blocks.iter()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|i| match i {
            Instruction::Alloca { dest, .. } => Some(dest.0),
            _ => None,
        })
        .collect();

    // Count uses per value
    let mut use_counts: Vec<(u32, u32, u32)> = Vec::new(); // (value_id, use_count, interval_start)
    for iv in &liveness.intervals {
        if alloca_set.contains(&iv.value_id) { continue; }
        if iv.end <= iv.start { continue; }

        // Check if this value spans any call
        let start_idx = liveness.call_points.partition_point(|&cp| cp < iv.start);
        let spans_call = start_idx < liveness.call_points.len()
            && liveness.call_points[start_idx] <= iv.end;
        if !spans_call { continue; }

        // Count uses of this value across all blocks
        let mut uses = 0u32;
        let vid = iv.value_id;
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst_uses_value(inst, vid) {
                    uses += 1;
                }
            }
        }

        if uses >= 3 { // Only split values used at least 3 times
            use_counts.push((iv.value_id, uses, iv.start));
        }
    }

    // Sort by use count descending
    use_counts.sort_by(|a, b| b.1.cmp(&a.1));

    // Step 3: Split top-N values
    let mut splits = 0;
    let mut next_val = func.next_value_id;

    for &(val_id, _uses, _) in use_counts.iter().take(max_splits) {
        // Find which blocks contain Calls that this value spans
        // For simplicity, work block-by-block: for each block containing a Call
        // where the value is live, insert Store before and Load after.

        // Determine the value's type by finding its definition
        let val_type = find_value_type(func, val_id);
        if val_type.is_none() { continue; }
        let val_type = val_type.unwrap();
        if val_type.is_float() || val_type.is_long_double() || val_type.is_128bit() {
            continue; // Skip complex types
        }

        // Create an alloca for the spill slot in the entry block
        let alloca_val = Value(next_val);
        next_val += 1;

        let alloca_inst = Instruction::Alloca {
            dest: alloca_val,
            ty: val_type,
            size: val_type.size(),
            align: 0,
            volatile: false,
        };

        // Insert alloca at the beginning of the entry block
        func.blocks[0].instructions.insert(0, alloca_inst);
        if !func.blocks[0].source_spans.is_empty() {
            func.blocks[0].source_spans.insert(0, crate::common::source::Span::dummy());
        }

        // For each block, find Calls and insert Store/Load pairs
        let mut any_split = false;
        for block_idx in 0..func.blocks.len() {
            let block = &func.blocks[block_idx];

            // Check if this block has any Call instructions
            let call_indices: Vec<usize> = block.instructions.iter().enumerate()
                .filter(|(_, inst)| matches!(inst,
                    Instruction::Call { .. } | Instruction::CallIndirect { .. }))
                .map(|(idx, _)| idx)
                .collect();

            if call_indices.is_empty() { continue; }

            // Check if the value is used in this block
            let val_used_in_block = block.instructions.iter().any(|inst| {
                inst_uses_value(inst, val_id)
            });
            if !val_used_in_block { continue; }

            // Insert Store/Load pairs around each Call in this block
            // Work backwards to preserve indices
            let mut new_instructions = Vec::with_capacity(block.instructions.len() + call_indices.len() * 2);
            let mut current_val = val_id; // Track which SSA value to use

            for (inst_idx, inst) in func.blocks[block_idx].instructions.iter().enumerate() {
                let is_call = matches!(inst,
                    Instruction::Call { .. } | Instruction::CallIndirect { .. });

                if is_call {
                    // Before the call: store current value to spill slot
                    new_instructions.push(Instruction::Store {
                        val: Operand::Value(Value(current_val)),
                        ptr: alloca_val,
                        ty: val_type,
                        seg_override: AddressSpace::Default,
                    });

                    // The call itself
                    new_instructions.push(inst.clone());

                    // After the call: reload from spill slot into new SSA value
                    let new_val = Value(next_val);
                    next_val += 1;
                    new_instructions.push(Instruction::Load {
                        dest: new_val,
                        ptr: alloca_val,
                        ty: val_type,
                        seg_override: AddressSpace::Default,
                    });

                    current_val = new_val.0;
                    any_split = true;
                } else {
                    // Replace uses of the original value with current_val
                    if current_val != val_id {
                        let mut inst_clone = inst.clone();
                        replace_value_uses(&mut inst_clone, val_id, Value(current_val));
                        new_instructions.push(inst_clone);
                    } else {
                        new_instructions.push(inst.clone());
                    }
                }
            }

            // Also replace in terminator if needed
            if current_val != val_id {
                replace_terminator_uses(&mut func.blocks[block_idx].terminator, val_id, Value(current_val));
            }

            func.blocks[block_idx].instructions = new_instructions;
            // Reset source_spans (they'll be wrong after insertion, but codegen handles missing spans)
            func.blocks[block_idx].source_spans.clear();
        }

        if any_split {
            splits += 1;
        }
    }

    func.next_value_id = next_val;
    splits
}

/// Find the type of a value by looking at its defining instruction.
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
                Instruction::Select { dest, ty, .. } => {
                    if dest.0 == val_id { return Some(*ty); }
                }
                Instruction::Phi { dest, ty, .. } => {
                    if dest.0 == val_id { return Some(*ty); }
                }
                _ => {}
            }
        }
    }
    None
}

/// Check if an instruction uses a given value ID.
fn inst_uses_value(inst: &Instruction, val_id: u32) -> bool {
    let check = |op: &Operand| -> bool {
        matches!(op, Operand::Value(v) if v.0 == val_id)
    };
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => {
            check(lhs) || check(rhs)
        }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } | Instruction::Copy { src, .. } => {
            check(src)
        }
        Instruction::Store { val, ptr, .. } => check(val) || ptr.0 == val_id,
        Instruction::Load { ptr, .. } => ptr.0 == val_id,
        Instruction::GetElementPtr { base, offset, .. } => base.0 == val_id || check(offset),
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            info.args.iter().any(|a| check(a))
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            check(cond) || check(true_val) || check(false_val)
        }
        _ => false,
    }
}

/// Replace uses of old_val with new_val in an instruction's operands.
fn replace_value_uses(inst: &mut Instruction, old_id: u32, new_val: Value) {
    let replace = |op: &mut Operand| {
        if let Operand::Value(v) = op {
            if v.0 == old_id { *v = new_val; }
        }
    };
    match inst {
        Instruction::BinOp { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => {
            replace(lhs); replace(rhs);
        }
        Instruction::UnaryOp { src, .. } | Instruction::Cast { src, .. } | Instruction::Copy { src, .. } => {
            replace(src);
        }
        Instruction::Store { val, ptr, .. } => {
            replace(val);
            if ptr.0 == old_id { *ptr = new_val; }
        }
        Instruction::Load { ptr, .. } => {
            if ptr.0 == old_id { *ptr = new_val; }
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            if base.0 == old_id { *base = new_val; }
            replace(offset);
        }
        Instruction::Call { info, .. } | Instruction::CallIndirect { info, .. } => {
            for a in &mut info.args { replace(a); }
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            replace(cond); replace(true_val); replace(false_val);
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming { replace(op); }
        }
        _ => {}
    }
}

/// Replace uses of old_val in a block terminator.
fn replace_terminator_uses(term: &mut Terminator, old_id: u32, new_val: Value) {
    let replace = |op: &mut Operand| {
        if let Operand::Value(v) = op {
            if v.0 == old_id { *v = new_val; }
        }
    };
    match term {
        Terminator::Return(Some(op)) => replace(op),
        Terminator::CondBranch { cond, .. } => replace(cond),
        Terminator::Switch { val, .. } => replace(val),
        _ => {}
    }
}
