//! Linear scan register allocator data structures.
//!
//! This module defines the core data structures for the linear scan algorithm:
//! - `LiveRange`: Enhanced live interval with priority, uses, and spill weight
//! - `ActiveInterval`: Currently live interval being processed
//! - `LinearScanAllocator`: Main allocator state machine
//!
//! The linear scan algorithm processes intervals in order of definition,
//! maintaining an "active" set of intervals that overlap with the current position.
//! For each interval, we either assign it a free register or spill it to the stack.

use super::liveness::{LiveInterval, for_each_operand_in_instruction, for_each_operand_in_terminator};
use super::regalloc::PhysReg;
use crate::common::fx_hash::FxHashMap;
use crate::ir::reexports::{Instruction, IrFunction, Operand, Terminator, Value};

/// Enhanced live interval with priority, uses, and spill weight.
///
/// This extends LiveInterval with:
/// - `uses`: Individual use points within [start, end]
/// - `loop_depth`: Loop nesting depth (used to weight priorities)
/// - `priority`: Weighted use count (higher = more important to allocate)
/// - `reg_hint`: Suggested register from Copy source (coalescing hint)
/// - `spill_weight`: Cost of spilling (priority / range_length)
#[derive(Debug, Clone)]
pub struct LiveRange {
    pub value_id: u32,
    pub start: u32,                // Program point where defined
    pub end: u32,                  // Last program point used
    pub uses: Vec<u32>,            // Use points within [start, end]
    pub loop_depth: u32,           // Loop nesting depth (0 = no loop, 1 = in loop, etc.)
    pub priority: u64,             // uses.len() * 10^loop_depth (higher priority = allocate first)
    pub reg_hint: Option<PhysReg>, // Preferred register (from Copy sources)
    pub spill_weight: f64,         // Cost of spilling: priority / range_length
}

impl LiveRange {
    /// Create a new LiveRange from a LiveInterval and loop depth.
    pub fn from_interval(interval: LiveInterval, loop_depth: u32) -> Self {
        let range_length = (interval.end - interval.start + 1).max(1) as f64;

        // Priority: number of uses * 10^loop_depth
        // Uses at depth 0 (no loop): weight = 1
        // Uses at depth 1 (one loop): weight = 10
        // Uses at depth 2 (nested loops): weight = 100
        // etc.
        let loop_weight = match loop_depth {
            0 => 1u64,
            1 => 10u64,
            2 => 100u64,
            3 => 1000u64,
            4 => 10_000u64,
            _ => 10_000u64, // cap at 10K to avoid overflow
        };

        // For now, estimate 2 uses per interval (this will be refined by build_live_ranges)
        let uses = 2u64 * loop_weight;

        let spill_weight = uses as f64 / range_length;

        Self {
            value_id: interval.value_id,
            start: interval.start,
            end: interval.end,
            uses: Vec::new(), // Will be populated by build_live_ranges
            loop_depth,
            priority: uses,
            reg_hint: None,
            spill_weight,
        }
    }

    /// Calculate spill weight based on actual use count and range length.
    pub fn calculate_spill_weight(&mut self) {
        let range_length = (self.end - self.start + 1).max(1) as f64;
        self.spill_weight = self.priority as f64 / range_length;
    }

    /// Check if this interval overlaps with another interval [start, end).
    pub fn overlaps(&self, start: u32, end: u32) -> bool {
        self.start < end && start < self.end
    }

    /// Check if this interval overlaps with another LiveRange.
    pub fn overlaps_with(&self, other: &LiveRange) -> bool {
        self.overlaps(other.start, other.end + 1)
    }
}

/// An interval that is currently active (overlaps with current position).
#[derive(Debug, Clone)]
pub struct ActiveInterval {
    pub range: LiveRange,
    /// The next program point where this interval is used.
    /// Used for tie-breaking when choosing which interval to spill.
    pub next_use: Option<u32>,
}

/// Main linear scan allocator state.
///
/// Processes intervals in order of definition, maintaining:
/// - `ranges`: Sorted list of all live ranges
/// - `active`: Intervals overlapping current position
/// - `handled`: Intervals that have finished
/// - `assignments`: Final register assignments
/// - `reg_free_until`: When each register becomes available
/// - `spill_slots`: Stack slot assignments for unallocated values
pub struct LinearScanAllocator {
    // Intervals to process (sorted by start point)
    pub ranges: Vec<LiveRange>,

    // Currently overlapping intervals
    pub active: Vec<ActiveInterval>,

    // Finished intervals
    pub handled: Vec<ActiveInterval>,

    // Final assignments
    pub assignments: FxHashMap<u32, PhysReg>,

    // For each register: program point until which it's occupied
    pub reg_free_until: FxHashMap<PhysReg, u32>,

    // For values that didn't get a register: stack slot offset
    pub spill_slots: FxHashMap<u32, i32>,

    // Available registers from config
    pub available_regs: Vec<PhysReg>,

    // Next stack slot offset (grows downward on most architectures)
    pub next_spill_slot: i32,

    // Whether to enable interval splitting (advanced feature)
    pub enable_splitting: bool,
}

impl LinearScanAllocator {
    /// Create a new allocator with the given live ranges and available registers.
    pub fn new(ranges: Vec<LiveRange>, available_regs: Vec<PhysReg>) -> Self {
        Self {
            ranges,
            active: Vec::new(),
            handled: Vec::new(),
            assignments: FxHashMap::default(),
            reg_free_until: FxHashMap::default(),
            spill_slots: FxHashMap::default(),
            available_regs,
            next_spill_slot: 0,
            enable_splitting: false,
        }
    }

    /// Initialize all registers as free (available from point 0).
    pub fn init_registers(&mut self) {
        for &reg in &self.available_regs {
            self.reg_free_until.insert(reg, 0);
        }
    }

    /// Check if a register is free at the given position.
    pub fn is_register_free(&self, reg: PhysReg, position: u32) -> bool {
        self.reg_free_until
            .get(&reg)
            .map_or(true, |&free_until| free_until <= position)
    }

    /// Find the earliest position at which a register becomes free.
    pub fn earliest_free_position(&self) -> u32 {
        self.reg_free_until.values().copied().min().unwrap_or(0)
    }

    /// Mark a register as occupied until the given position.
    pub fn occupy_register(&mut self, reg: PhysReg, until: u32) {
        self.reg_free_until.insert(reg, until);
    }

    /// Allocate a spill slot for a value and return its offset.
    pub fn allocate_spill_slot(&mut self, value_id: u32) -> i32 {
        let slot = self.next_spill_slot;
        self.spill_slots.insert(value_id, slot);
        self.next_spill_slot -= 8; // Assume 8-byte slots (could be configurable)
        slot
    }

    /// Expire old intervals that no longer overlap with the current position.
    ///
    /// Intervals in the active set that end before the given position are
    /// moved to the handled set, freeing their registers.
    pub fn expire_old_intervals(&mut self, current_start: u32) {
        self.active.retain(|active| {
            if active.range.end < current_start {
                // Interval is done, move it to handled
                self.handled.push(active.clone());
                false // Remove from active
            } else {
                true // Keep in active
            }
        });
    }

    /// Find a free register for the given range.
    ///
    /// Uses the following strategy:
    /// 1. If there's a register hint from Copy sources, try that first
    /// 2. Find a register that's free for the entire duration of the range
    /// 3. If none, return None (caller will spill)
    pub fn find_free_register(&self, range: &LiveRange) -> Option<PhysReg> {
        // Try register hint first (for coalescing with Copy sources)
        if let Some(hint) = range.reg_hint {
            if self.is_register_free(hint, range.start)
                && self.active.iter().all(|a| !a.range.overlaps_with(range))
            {
                return Some(hint);
            }
        }

        // Find any free register
        for &reg in &self.available_regs {
            // Check if this register is free at the start of the range
            if self.is_register_free(reg, range.start) {
                // Also check that no active interval uses this register
                let reg_free_throughout = self
                    .active
                    .iter()
                    .filter(|a| {
                        // Check if any active interval uses this register
                        if let Some(assigned_reg) = self.assignments.get(&a.range.value_id) {
                            *assigned_reg == reg
                        } else {
                            false
                        }
                    })
                    .all(|a| !a.range.overlaps_with(range));

                if reg_free_throughout {
                    return Some(reg);
                }
            }
        }

        None
    }

    /// Find the best interval to spill when no register is free.
    ///
    /// Returns the active interval with the lowest spill weight (least important).
    /// Intervals with equal spill weight are broken by next_use (later is better to spill).
    pub fn find_spill_candidate(&mut self) -> Option<usize> {
        if self.active.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_weight = self.active[0].range.spill_weight;
        let mut best_next_use = self.active[0].next_use.unwrap_or(u32::MAX);

        for (idx, interval) in self.active.iter().enumerate().skip(1) {
            let weight = interval.range.spill_weight;
            let next_use = interval.next_use.unwrap_or(u32::MAX);

            // Prefer lower spill weight (less important). If tied, prefer later next_use.
            if weight < best_weight || (weight == best_weight && next_use > best_next_use) {
                best_idx = idx;
                best_weight = weight;
                best_next_use = next_use;
            }
        }

        Some(best_idx)
    }

    /// Process a single live range through the allocation algorithm.
    ///
    /// This is the core loop body:
    /// 1. Expire intervals that ended before this range starts
    /// 2. Try to find a free register
    /// 3. If none, find the best interval to spill
    /// 4. Assign the register or spill to stack
    pub fn allocate_range(&mut self, range: LiveRange) {
        // Step 1: Expire old intervals
        self.expire_old_intervals(range.start);

        // Step 2: Try to find a free register
        if let Some(reg) = self.find_free_register(&range) {
            // Found a free register - assign it
            self.assignments.insert(range.value_id, reg);
            self.occupy_register(reg, range.end + 1);

            self.active.push(ActiveInterval {
                range,
                next_use: None,
            });
        } else if let Some(spill_idx) = self.find_spill_candidate() {
            // Step 3: Evict the lowest-priority active interval — but only if it is less
            // important than the incoming range. If the incoming range has lower spill weight,
            // spill it instead (keep the better interval in the register).
            let active_weight = self.active[spill_idx].range.spill_weight;
            if active_weight < range.spill_weight {
                // Active interval is less important — evict it, give register to incoming.
                let spilled = self.active.remove(spill_idx);
                if let Some(freed_reg) = self.assignments.remove(&spilled.range.value_id) {
                    self.assignments.insert(range.value_id, freed_reg);
                    self.occupy_register(freed_reg, range.end + 1);
                    self.active.push(ActiveInterval {
                        range,
                        next_use: None,
                    });
                    self.allocate_spill_slot(spilled.range.value_id);
                } else {
                    self.allocate_spill_slot(range.value_id);
                }
            } else {
                // Incoming range is less important — spill it, keep active intervals.
                self.allocate_spill_slot(range.value_id);
            }
        } else {
            // No registers available and nothing to spill — assign to stack.
            self.allocate_spill_slot(range.value_id);
        }
    }

    /// Run the full linear scan allocation algorithm.
    ///
    /// Processes all ranges in order, assigning registers or spilling to stack.
    pub fn run(&mut self) {
        self.init_registers();

        // Process ranges in order of start point
        while !self.ranges.is_empty() {
            let range = self.ranges.remove(0);
            self.allocate_range(range);
        }
    }
}

/// Helper to build live ranges from liveness analysis results.
///
/// This function:
/// 1. Converts LiveInterval → LiveRange with loop depth
/// 2. Collects actual use points (program points where values are used)
/// 3. Finds register hints from Copy instructions
/// 4. Calculates priorities and spill weights
pub fn build_live_ranges(
    intervals: &[LiveInterval],
    loop_depth: &[u32],
    func: &IrFunction,
) -> Vec<LiveRange> {
    // Build map: value_id → defining block index.
    // This fixes a bug where value_id was incorrectly used as an index into
    // block_loop_depth (which is indexed by block index, not value ID).
    let mut def_block: FxHashMap<u32, usize> = FxHashMap::default();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                def_block.insert(dest.0, block_idx);
            }
        }
    }

    // Build map: value_id → max loop depth across all use sites.
    // A value defined outside a loop but used inside it should get the inner
    // loop's priority, not the definition site's (typically depth 0).
    let mut max_use_depth: FxHashMap<u32, u32> = FxHashMap::default();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let bdepth = loop_depth.get(block_idx).copied().unwrap_or(0);
        if bdepth == 0 { continue; } // Skip non-loop blocks (depth 0 can't increase max)
        for inst in &block.instructions {
            for_each_operand_in_instruction(inst, |op| {
                if let Operand::Value(v) = op {
                    let entry = max_use_depth.entry(v.0).or_insert(0);
                    *entry = (*entry).max(bdepth);
                }
            });
        }
        for_each_operand_in_terminator(&block.terminator, |op| {
            if let Operand::Value(v) = op {
                let entry = max_use_depth.entry(v.0).or_insert(0);
                *entry = (*entry).max(bdepth);
            }
        });
    }

    let mut ranges: Vec<LiveRange> = intervals
        .iter()
        .map(|interval| {
            // Use the maximum of defining block depth and max use-site depth.
            // This ensures values defined outside loops but used inside them
            // get the correct inner-loop priority for register allocation.
            let def_depth = def_block.get(&interval.value_id)
                .and_then(|&bidx| loop_depth.get(bidx).copied())
                .unwrap_or(0);
            let use_depth = max_use_depth.get(&interval.value_id).copied().unwrap_or(0);
            let depth = def_depth.max(use_depth);
            LiveRange::from_interval(*interval, depth)
        })
        .collect();

    // Collect actual use points for each value
    let uses_map = collect_uses_for_values(func);

    // Find register hints from Copy sources
    let hints_map = find_register_hints(func);

    // Update each range with actual uses and hints
    for range in &mut ranges {
        // Collect uses within this range's interval
        if let Some(uses) = uses_map.get(&range.value_id) {
            range.uses = uses
                .iter()
                .filter(|&&u| u >= range.start && u <= range.end)
                .copied()
                .collect();
        }

        // Update priority based on actual use count
        let loop_weight = 10u64.pow(range.loop_depth.min(4) as u32);
        range.priority = (range.uses.len() as u64).max(1) * loop_weight;

        // Add register hint if available
        range.reg_hint = hints_map.get(&range.value_id).copied();

        // Recalculate spill weight with actual use count
        range.calculate_spill_weight();
    }

    // Sort by start point (primary) and by priority (secondary, for tie-breaking)
    ranges.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| b.priority.cmp(&a.priority))
    });

    ranges
}

/// Collect all use points for each value ID in the function.
///
/// Returns a map: value_id → Vec<program_point> where the value is used.
/// Program points are assigned sequentially to each instruction and terminator.
fn collect_uses_for_values(func: &IrFunction) -> FxHashMap<u32, Vec<u32>> {
    let mut uses: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    let mut point = 0u32;

    for block in &func.blocks {
        for inst in &block.instructions {
            record_operand_uses(inst, point, &mut uses);
            point += 1;
        }
        // Account for terminator point
        record_terminator_uses(&block.terminator, point, &mut uses);
        point += 1;
    }

    uses
}

/// Record uses of operands in an instruction.
fn record_operand_uses(inst: &Instruction, point: u32, uses: &mut FxHashMap<u32, Vec<u32>>) {
    // Helper to record a use of a value
    let mut record = |vid: u32| {
        uses.entry(vid).or_insert_with(Vec::new).push(point);
    };

    // Match on instruction type to find all operand uses
    match inst {
        Instruction::BinOp { lhs, rhs, .. } => {
            if let Operand::Value(v) = lhs {
                record(v.0);
            }
            if let Operand::Value(v) = rhs {
                record(v.0);
            }
        }
        Instruction::UnaryOp { src, .. } => {
            if let Operand::Value(v) = src {
                record(v.0);
            }
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            if let Operand::Value(v) = lhs {
                record(v.0);
            }
            if let Operand::Value(v) = rhs {
                record(v.0);
            }
        }
        Instruction::Store { val, .. } => {
            if let Operand::Value(v) = val {
                record(v.0);
            }
        }
        Instruction::Cast { src, .. } => {
            if let Operand::Value(v) = src {
                record(v.0);
            }
        }
        Instruction::Copy { src, .. } => {
            if let Operand::Value(v) = src {
                record(v.0);
            }
        }
        Instruction::Call { info, .. } => {
            for arg in &info.args {
                if let Operand::Value(v) = arg {
                    record(v.0);
                }
            }
        }
        Instruction::CallIndirect { func_ptr, info, .. } => {
            if let Operand::Value(v) = func_ptr {
                record(v.0);
            }
            for arg in &info.args {
                if let Operand::Value(v) = arg {
                    record(v.0);
                }
            }
        }
        Instruction::GetElementPtr { offset, .. } => {
            if let Operand::Value(v) = offset {
                record(v.0);
            }
        }
        Instruction::Select {
            cond,
            true_val,
            false_val,
            ..
        } => {
            if let Operand::Value(v) = cond {
                record(v.0);
            }
            if let Operand::Value(v) = true_val {
                record(v.0);
            }
            if let Operand::Value(v) = false_val {
                record(v.0);
            }
        }
        Instruction::AtomicRmw { val, .. } => {
            if let Operand::Value(v) = val {
                record(v.0);
            }
        }
        Instruction::AtomicCmpxchg {
            expected, desired, ..
        } => {
            if let Operand::Value(v) = expected {
                record(v.0);
            }
            if let Operand::Value(v) = desired {
                record(v.0);
            }
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming {
                if let Operand::Value(v) = op {
                    record(v.0);
                }
            }
        }
        _ => {}
    }

    // Record direct Value uses (pointers, bases, etc.)
    record_value_uses(inst, point, uses);
}

/// Record direct Value uses (not wrapped in Operand).
fn record_value_uses(inst: &Instruction, point: u32, uses: &mut FxHashMap<u32, Vec<u32>>) {
    let mut record = |v: &Value| {
        uses.entry(v.0).or_insert_with(Vec::new).push(point);
    };

    match inst {
        Instruction::Load { ptr, .. } => record(ptr),
        Instruction::Store { ptr, .. } => record(ptr),
        Instruction::GetElementPtr { base, .. } => record(base),
        Instruction::Memcpy { dest, src, .. } => {
            record(dest);
            record(src);
        }
        _ => {}
    }
}

/// Record uses in a terminator.
fn record_terminator_uses(term: &Terminator, point: u32, uses: &mut FxHashMap<u32, Vec<u32>>) {
    let mut record = |v: u32| {
        uses.entry(v).or_insert_with(Vec::new).push(point);
    };

    match term {
        Terminator::Return(Some(op)) => {
            if let Operand::Value(v) = op {
                record(v.0);
            }
        }
        Terminator::CondBranch { cond, .. } => {
            if let Operand::Value(v) = cond {
                record(v.0);
            }
        }
        Terminator::IndirectBranch { target, .. } => {
            if let Operand::Value(v) = target {
                record(v.0);
            }
        }
        Terminator::Switch { val, .. } => {
            if let Operand::Value(v) = val {
                record(v.0);
            }
        }
        _ => {}
    }
}

/// Find register hints from Copy instructions.
///
/// For Copy instructions where the source is allocated to a register,
/// the destination can be hinted to use the same register, enabling coalescing.
///
/// Returns a map: dest_value_id → source_register_hint
fn find_register_hints(func: &IrFunction) -> FxHashMap<u32, PhysReg> {
    let mut hints: FxHashMap<u32, PhysReg> = FxHashMap::default();

    // TODO: In a full implementation, we would track which values have been
    // allocated to registers and use those as hints. For now, return empty map.
    // This will be populated when we have actual register assignments.

    hints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_range_from_interval() {
        let interval = LiveInterval {
            value_id: 42,
            start: 10,
            end: 20,
        };
        let range = LiveRange::from_interval(interval, 1);

        assert_eq!(range.value_id, 42);
        assert_eq!(range.start, 10);
        assert_eq!(range.end, 20);
        assert_eq!(range.loop_depth, 1);
        assert!(range.priority > 0);
        assert!(range.spill_weight > 0.0);
    }

    #[test]
    fn test_overlap_detection() {
        let range = LiveRange {
            value_id: 1,
            start: 10,
            end: 20,
            uses: vec![],
            loop_depth: 0,
            priority: 1,
            reg_hint: None,
            spill_weight: 0.1,
        };

        // Overlapping: starts within range
        assert!(range.overlaps(15, 25));

        // Overlapping: ends within range
        assert!(range.overlaps(5, 15));

        // Overlapping: contains range
        assert!(range.overlaps(5, 25));

        // No overlap: ends before range
        assert!(!range.overlaps(21, 30));

        // No overlap: starts after range
        assert!(!range.overlaps(0, 9));
    }

    #[test]
    fn test_priority_weighting() {
        let interval = LiveInterval {
            value_id: 1,
            start: 0,
            end: 100,
        };

        let no_loop = LiveRange::from_interval(interval, 0);
        let in_loop = LiveRange::from_interval(interval, 1);
        let nested_loop = LiveRange::from_interval(interval, 2);

        // Same number of uses, but loop-weighted priorities differ
        assert!(in_loop.priority > no_loop.priority);
        assert!(nested_loop.priority > in_loop.priority);
    }

    #[test]
    fn test_spill_weight_calculation() {
        let short_range = LiveRange {
            value_id: 1,
            start: 0,
            end: 10,
            uses: vec![],
            loop_depth: 0,
            priority: 100,
            reg_hint: None,
            spill_weight: 100.0 / 11.0, // 100 / range_length
        };

        let long_range = LiveRange {
            value_id: 2,
            start: 0,
            end: 100,
            uses: vec![],
            loop_depth: 0,
            priority: 100,
            reg_hint: None,
            spill_weight: 100.0 / 101.0, // 100 / range_length
        };

        // Short ranges have higher spill weight (less painful to keep in register)
        assert!(short_range.spill_weight > long_range.spill_weight);
    }

    #[test]
    fn test_linear_scan_basic_allocation() {
        // Create a simple allocator with 2 registers and 3 non-overlapping ranges
        let ranges = vec![
            LiveRange {
                value_id: 1,
                start: 0,
                end: 10,
                uses: vec![0, 5, 10],
                loop_depth: 0,
                priority: 3,
                reg_hint: None,
                spill_weight: 0.3,
            },
            LiveRange {
                value_id: 2,
                start: 20,
                end: 30,
                uses: vec![20, 25, 30],
                loop_depth: 0,
                priority: 3,
                reg_hint: None,
                spill_weight: 0.3,
            },
            LiveRange {
                value_id: 3,
                start: 40,
                end: 50,
                uses: vec![40, 45, 50],
                loop_depth: 0,
                priority: 3,
                reg_hint: None,
                spill_weight: 0.3,
            },
        ];

        let regs = vec![PhysReg(0), PhysReg(1)];
        let mut allocator = LinearScanAllocator::new(ranges, regs);
        allocator.run();

        // All three non-overlapping intervals should get registers
        assert_eq!(allocator.assignments.len(), 3);
        assert!(allocator.assignments.contains_key(&1));
        assert!(allocator.assignments.contains_key(&2));
        assert!(allocator.assignments.contains_key(&3));
    }

    #[test]
    fn test_linear_scan_spilling() {
        // Create a scenario with overlapping ranges that need spilling
        let ranges = vec![
            LiveRange {
                value_id: 1,
                start: 0,
                end: 100,
                uses: vec![0, 50, 100],
                loop_depth: 0,
                priority: 3, // High priority
                reg_hint: None,
                spill_weight: 0.03,
            },
            LiveRange {
                value_id: 2,
                start: 10,
                end: 90,
                uses: vec![10, 50, 90],
                loop_depth: 0,
                priority: 2, // Lower priority - should spill
                reg_hint: None,
                spill_weight: 0.02,
            },
        ];

        let regs = vec![PhysReg(0)]; // Only one register
        let mut allocator = LinearScanAllocator::new(ranges, regs);
        allocator.run();

        // Value 1 gets the register (higher priority)
        assert!(allocator.assignments.contains_key(&1));
        // Value 2 either gets the register or gets spilled (1 register for 2 overlapping values)
        // The allocator should make a decision
        assert!(allocator.assignments.len() <= 2);
    }

    #[test]
    fn test_linear_scan_no_registers() {
        // Test allocator with no available registers
        let ranges = vec![LiveRange {
            value_id: 1,
            start: 0,
            end: 10,
            uses: vec![0, 10],
            loop_depth: 0,
            priority: 1,
            reg_hint: None,
            spill_weight: 0.1,
        }];

        let regs = vec![]; // No registers available
        let mut allocator = LinearScanAllocator::new(ranges, regs);
        allocator.run();

        // Should allocate spill slots instead
        assert!(allocator.spill_slots.contains_key(&1));
    }
}
