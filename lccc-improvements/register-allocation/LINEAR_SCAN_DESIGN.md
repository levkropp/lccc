# Linear Scan Register Allocation: Design Specification

## Overview

This document specifies the improved linear scan register allocator for LCCC Phase 1. The goal is to achieve **3-4x performance improvement** by better allocating IR values to registers instead of spilling to stack.

## Algorithm: Classic Linear Scan with Interval Splitting

### Reference

Based on LLVM's RegAllocLinearScan and the original Poletto/Sarkar paper "Linear Scan Register Allocation" (2000).

### Key Improvements Over Current Allocator

| Feature | Current | New | Impact |
|---------|---------|-----|--------|
| **Interval Splitting** | No (all-or-nothing) | Yes (split at spill points) | 2-3x fewer spills |
| **Register Coalescing** | No | Yes (merge copy sources/dests) | Eliminates redundant moves |
| **Spill Costs** | None (greedy) | Loop-weighted | Better register selection |
| **Dead Code** | None | Remove before allocation | Fewer live ranges |
| **Eligibility** | ~5% | ~40-50% (with float handling) | More values get registers |

## Data Structures

### 1. LiveRange (Enhanced LiveInterval)

```rust
pub struct LiveRange {
    /// Value being allocated
    pub value_id: u32,
    
    /// [start, end] program points
    pub start: u32,
    pub end: u32,
    
    /// Use points within this interval (sorted)
    pub uses: Vec<u32>,
    
    /// Loop nesting depth (from liveness analysis)
    pub loop_depth: u32,
    
    /// Weighted use count (uses * 10^loop_depth)
    pub priority: u64,
    
    /// Register preference (hint from copy sources)
    pub reg_hint: Option<PhysReg>,
    
    /// Spill weight: cost of spilling this range
    /// = num_uses * 10^loop_depth / (range_length)
    pub spill_weight: f64,
}
```

### 2. ActiveInterval

```rust
pub struct ActiveInterval {
    /// Which physical register is assigned
    pub reg: PhysReg,
    
    /// Live range for this allocation
    pub range: LiveRange,
    
    /// Next use point of this value
    pub next_use: Option<u32>,
}
```

### 3. AllocatorState

```rust
pub struct LinearScanAllocator {
    /// All live ranges, sorted by start point
    pub ranges: Vec<LiveRange>,
    
    /// Currently active intervals (those whose ranges overlap current position)
    pub active: Vec<ActiveInterval>,
    
    /// Intervals that have finished (no longer overlap current position)
    pub handled: Vec<ActiveInterval>,
    
    /// Assignments: value_id → physical register
    pub assignments: FxHashMap<u32, PhysReg>,
    
    /// Physical register → next free program point
    pub reg_free_until: FxHashMap<PhysReg, u32>,
    
    /// Spill slots allocated (value_id → stack offset)
    pub spill_slots: FxHashMap<u32, i32>,
    
    /// Available registers to allocate
    pub available_regs: Vec<PhysReg>,
    
    /// Whether to split intervals
    pub enable_splitting: bool,
}
```

## Algorithm: Main Loop

```rust
pub fn allocate(
    func: &IrFunction,
    config: &RegAllocConfig,
) -> RegAllocResult {
    let liveness = compute_live_intervals(func);
    
    // Step 1: Build live ranges with priorities and hints
    let mut ranges = build_live_ranges(&liveness, func);
    ranges.sort_by_key(|r| r.start);
    
    // Step 2: Initialize allocator state
    let mut state = LinearScanAllocator::new(config.available_regs.clone());
    
    // Step 3: Main allocation loop
    for range in &ranges {
        state.allocate_range(range, &config);
    }
    
    // Step 4: Assign spill slots to unallocated values
    state.assign_spill_slots(&liveness);
    
    // Return results
    RegAllocResult {
        assignments: state.assignments,
        used_regs: state.get_used_regs(),
        liveness: Some(liveness),
    }
}
```

## Step 1: Build Live Ranges

```rust
fn build_live_ranges(liveness: &LivenessResult, func: &IrFunction) -> Vec<LiveRange> {
    let mut ranges = Vec::new();
    
    for interval in &liveness.intervals {
        // Skip allocas and non-eligible values
        if should_skip_allocation(interval.value_id, func) {
            continue;
        }
        
        // Collect use points within [start, end]
        let uses = collect_uses(interval.value_id, interval.start, interval.end, func);
        
        if uses.is_empty() {
            continue;  // Dead value, skip
        }
        
        // Compute priority: use count weighted by loop depth
        let loop_depth = liveness.block_loop_depth[block_containing_value(interval.value_id)];
        let weight = 10_u64.pow(loop_depth.min(4) as u32);
        let priority = (uses.len() as u64) * weight;
        
        // Check for register hints from copy sources
        let reg_hint = find_reg_hint(interval.value_id, func, &state.assignments);
        
        // Spill weight: prefer to keep high-use values
        let range_length = (interval.end - interval.start).max(1);
        let spill_weight = (uses.len() as f64) * (weight as f64) / (range_length as f64);
        
        ranges.push(LiveRange {
            value_id: interval.value_id,
            start: interval.start,
            end: interval.end,
            uses,
            loop_depth,
            priority,
            reg_hint,
            spill_weight,
        });
    }
    
    ranges
}
```

## Step 2: Allocate Range

```rust
fn allocate_range(&mut self, range: &LiveRange, config: &RegAllocConfig) {
    // Step 2a: Expire finished intervals
    self.expire_old_intervals(range.start);
    
    // Step 2b: Try to find a free register
    if let Some(reg) = self.find_free_register(range) {
        self.assign_register(range, reg);
        return;
    }
    
    // Step 2c: No free register, must spill
    if let Some(to_spill) = self.find_best_spill_candidate(range) {
        // Split the spilled interval and continue
        if self.enable_splitting {
            self.split_interval_at_first_use(to_spill, range.start);
        } else {
            self.unassign_register(to_spill);
        }
        self.assign_register(range, to_spill.reg);
    } else {
        // Spill this range to stack
        self.allocate_spill_slot(range.value_id);
    }
}
```

### 2a: Expire Old Intervals

```rust
fn expire_old_intervals(&mut self, current_start: u32) {
    // Remove intervals whose ends are before current start
    self.active.retain(|ival| {
        if ival.range.end < current_start {
            self.handled.push(ival.clone());
            false
        } else {
            true
        }
    });
}
```

### 2b: Find Free Register

```rust
fn find_free_register(&self, range: &LiveRange) -> Option<PhysReg> {
    // Strategy: prefer register hints, then callee-saved, then caller-saved
    
    // First: try the hinted register
    if let Some(hint) = range.reg_hint {
        if self.is_register_free(hint, range) {
            return Some(hint);
        }
    }
    
    // Second: prefer registers that are already used (minimize prologue/epilogue)
    for &reg in &self.available_regs {
        if self.is_register_free(reg, range) && self.is_register_already_used(reg) {
            return Some(reg);
        }
    }
    
    // Third: use any free register
    for &reg in &self.available_regs {
        if self.is_register_free(reg, range) {
            return Some(reg);
        }
    }
    
    None
}

fn is_register_free(&self, reg: PhysReg, range: &LiveRange) -> bool {
    if let Some(&free_until) = self.reg_free_until.get(&reg) {
        free_until <= range.start
    } else {
        true
    }
}
```

### 2c: Find Best Spill Candidate

```rust
fn find_best_spill_candidate(&self, range: &LiveRange) -> Option<&ActiveInterval> {
    // Choose the active interval with lowest spill weight
    // (prefer to spill low-use values, keep high-use values)
    
    self.active.iter().min_by(|a, b| {
        a.range.spill_weight.partial_cmp(&b.range.spill_weight).unwrap()
    })
}

fn split_interval_at_first_use(&mut self, to_spill: &ActiveInterval, split_point: u32) {
    // Move range end from to_spill.range.end to split_point
    // Create a new range starting at the first use after split_point
    
    let first_use_after = to_spill.range.uses.iter()
        .find(|&&u| u >= split_point)
        .copied();
    
    if let Some(next_use) = first_use_after {
        // Spill from split_point to next_use
        self.allocate_spill_slot(to_spill.range.value_id);
        // Keep assignment from split_point to next_use on the register
    }
}
```

## Step 3: Assign Spill Slots

```rust
fn assign_spill_slots(&mut self, liveness: &LivenessResult) {
    // Unallocated values get stack slots
    // Use liveness information to pack slots (overlapping ranges share slots)
    
    for interval in &liveness.intervals {
        if !self.assignments.contains_key(&interval.value_id) {
            let slot_offset = self.allocate_stack_slot(interval);
            self.spill_slots.insert(interval.value_id, slot_offset);
        }
    }
}

fn allocate_stack_slot(&mut self, interval: &LiveInterval) -> i32 {
    // Find a free slot that doesn't overlap this interval's lifetime
    // For simplicity, just allocate sequential slots
    
    let slot_size = 8;  // 64-bit slots
    let slot_num = self.spill_slots.len() as i32;
    slot_num * slot_size
}
```

## Register Coalescing

After allocation, merge copy chains to eliminate redundant moves:

```rust
fn coalesce_copies(&mut self, func: &IrFunction) {
    // For each Copy instruction:
    // if src → reg1, dest → reg2, and reg1 == reg2,
    // merge the source and dest intervals (rerun allocation if beneficial)
    
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Copy { dest, src: Operand::Value(src_val) } = inst {
                let dest_reg = self.assignments.get(&dest.0);
                let src_reg = self.assignments.get(&src_val.0);
                
                if dest_reg == src_reg {
                    // Copy source and dest have same register: can coalesce
                    // Mark for copy elimination (backend handles this)
                }
            }
        }
    }
}
```

## Spill Cost Calculation

```rust
fn calculate_spill_cost(range: &LiveRange, loop_depth: u32) -> f64 {
    let base_weight = 10_u64.pow(loop_depth.min(4) as u32) as f64;
    let num_uses = range.uses.len() as f64;
    let range_length = (range.end - range.start) as f64;
    
    // Penalize long intervals with few uses
    num_uses * base_weight / range_length
}
```

Higher cost = more likely to get a register (don't spill).

## Implementation Plan: Week 2

### Phase 2a: Data Structure (3 hours)
- [ ] Create `ccc/src/backend/live_range.rs`
- [ ] Implement `LiveRange` struct with comparison traits
- [ ] Implement `ActiveInterval`, `LinearScanAllocator`

### Phase 2b: Helper Functions (3 hours)
- [ ] `build_live_ranges()` — convert LiveInterval → LiveRange
- [ ] `collect_uses()` — find use points
- [ ] `find_reg_hint()` — extract register preferences

### Phase 2c: Main Algorithm (6 hours)
- [ ] `expire_old_intervals()`
- [ ] `find_free_register()` with priority heuristics
- [ ] `find_best_spill_candidate()` with spill cost
- [ ] `allocate_range()` main loop

### Phase 2d: Integration (3 hours)
- [ ] Integrate into `regalloc.rs`
- [ ] Ensure `RegAllocResult` compatibility
- [ ] Test on simple functions

## Performance Expectations

### Current Baseline (32-variable function)
```
Stack frame: 11264 bytes
Register assignments: 0
```

### Expected After Implementation
```
Stack frame: 256 bytes (alignment padding only)
Register assignments: 28-32 (all eligible values)
Speedup: 3-4x
```

### Validation Targets
1. **Micro-benchmark:** 32-variable function → measure assembly quality
2. **Unit test:** verify register assignments on known functions
3. **Real-world:** SQLite benchmark → measure end-to-end performance

## Fallback Strategy

If issues arise during implementation:

1. **Phase 1 (Conservative):** Implement without splitting
   - Just allocation without interval splitting
   - Simpler, easier to debug
   - Still achieves 2x improvement

2. **Phase 2 (Splitting):** Add interval splitting once base works
   - Split at use points to reduce spills
   - Adds ~20% more complexity

3. **Phase 3 (Coalescing):** Add copy coalescing
   - Merge unnecessarily split intervals
   - Final optimization

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_simple_two_var_allocation() {
    // Create IR with 2 values
    // Allocate with 2 available registers
    // Verify both get registers
}

#[test]
fn test_spill_when_no_regs() {
    // Create IR with 10 values
    // Allocate with 2 available registers
    // Verify best 2 keep registers, others spill
}

#[test]
fn test_loop_weighted_priority() {
    // Create IR with values in/out of loops
    // Verify loop values get priority
}
```

### Integration Tests
```c
// test_32_vars.c
int compute(int a, int b, ..., int f2) {
    return a + b + ... + f2;
}
```

```bash
./ccc -S -O2 test_32_vars.c
# Compare register assignments
# Verify stack frame minimal
```

### Benchmark
```bash
./ccc -c sqlite3.c -O2
# Time compilation
# Compare assembly size
# Run benchmarks
```

## Success Criteria

- [ ] All values in 32-variable function allocated to registers
- [ ] Stack frame reduced from 11KB to <1KB
- [ ] No regression on other test files
- [ ] Compilation speed unchanged (or faster)
- [ ] 3-4x speedup on compute-heavy benchmarks
