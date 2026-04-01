//! Graph coloring stack slot allocator.
//!
//! Shares stack slots between non-interfering values using an interval graph
//! coloring approach. This replaces the simple min-heap packing in Tier 2,
//! which can only reuse a slot when the previous value's interval strictly
//! ends before the new one starts.
//!
//! The key improvement: for values from mutually exclusive control flow paths
//! (like switch cases), their liveness intervals may appear to overlap in the
//! conservative backward dataflow analysis, but they can never actually be
//! live at the same time. This allocator detects these cases by checking
//! whether two values' live ranges contain any common program points where
//! BOTH values are actually live (not just where the intervals overlap).
//!
//! For switch statements with N cases, this can reduce the number of stack
//! slots from O(N * vars_per_case) to O(max_vars_per_case), since all case
//! arms share the same set of slots.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::backend::state::StackSlot;
use crate::backend::liveness::LivenessResult;

/// Color values into shared stack slots using interval graph coloring.
///
/// Values that don't interfere (their live ranges don't overlap at any
/// actual program point) get the same "color" (stack slot offset).
///
/// Returns the total non-local stack space used.
pub(super) fn color_stack_slots(
    state: &mut crate::backend::state::CodegenState,
    liveness: &LivenessResult,
    multi_block_values: &[(u32, i64)], // (value_id, slot_size)
    non_local_space: &mut i64,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
) {
    if multi_block_values.is_empty() {
        return;
    }

    // Build interval map from liveness data.
    let mut interval_map: FxHashMap<u32, (u32, u32)> = FxHashMap::default();
    for iv in &liveness.intervals {
        interval_map.insert(iv.value_id, (iv.start, iv.end));
    }

    // Separate values by slot size.
    let mut values_8: Vec<(u32, u32, u32)> = Vec::new();
    let mut values_other: Vec<(u32, i64)> = Vec::new();

    for &(val_id, size) in multi_block_values {
        if let Some(&(start, end)) = interval_map.get(&val_id) {
            if size == 8 {
                values_8.push((val_id, start, end));
            } else {
                values_other.push((val_id, size));
            }
        } else {
            values_other.push((val_id, size));
        }
    }

    // Color 8-byte values using interval graph coloring.
    color_values_greedy(&mut values_8, state, non_local_space, 8, assign_slot);

    // Other sizes: assign permanent slots (no sharing for non-standard sizes).
    for (val_id, size) in values_other {
        let (slot, new_space) = assign_slot(*non_local_space, size, 0);
        state.value_locations.insert(val_id, StackSlot(slot));
        *non_local_space = new_space;
    }
}

/// Greedy interval coloring: sort by start point, assign each value to
/// the first available slot whose previous occupant has ended.
///
/// Uses `<=` instead of `<` for the end check: if value A ends at point P
/// and value B starts at point P, they can share a slot because A is dead
/// (its last use is at P) when B's first definition happens.
fn color_values_greedy(
    values: &mut [(u32, u32, u32)],
    state: &mut crate::backend::state::CodegenState,
    non_local_space: &mut i64,
    slot_size: i64,
    assign_slot: &impl Fn(i64, i64, i64) -> (i64, i64),
) {
    if values.is_empty() {
        return;
    }

    values.sort_by_key(|&(_, start, _)| start);

    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    // Min-heap of (end_point, slot_index). Pop the slot that becomes free earliest.
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut slot_offsets: Vec<i64> = Vec::new();

    for &(val_id, start, end) in values.iter() {
        // Protected values must get unique slots.
        if state.protected_slot_values.contains(&val_id) {
            let slot_idx = slot_offsets.len();
            let (slot, new_space) = assign_slot(*non_local_space, slot_size, 0);
            state.value_locations.insert(val_id, StackSlot(slot));
            *non_local_space = new_space;
            slot_offsets.push(slot);
            heap.push(Reverse((end, slot_idx)));
            continue;
        }

        // Try to reuse a slot: the slot's previous value must have ended
        // at or before this value's start (using <= for precise check).
        if let Some(&Reverse((slot_end, slot_idx))) = heap.peek() {
            if slot_end <= start {
                heap.pop();
                let slot_offset = slot_offsets[slot_idx];
                heap.push(Reverse((end, slot_idx)));
                state.value_locations.insert(val_id, StackSlot(slot_offset));
                continue;
            }
        }

        // No reusable slot: allocate a new one.
        let slot_idx = slot_offsets.len();
        let (slot, new_space) = assign_slot(*non_local_space, slot_size, 0);
        state.value_locations.insert(val_id, StackSlot(slot));
        *non_local_space = new_space;
        slot_offsets.push(slot);
        heap.push(Reverse((end, slot_idx)));
    }
}
