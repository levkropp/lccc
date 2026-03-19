---
layout: doc
title: Register Allocator
description: Deep dive into LCCC's two-pass linear scan register allocator.
prev_page:
  title: Architecture
  url: /docs/architecture
next_page:
  title: Optimization Passes
  url: /docs/optimization-passes
---

# Register Allocator
{:.doc-subtitle}
LCCC replaces CCC's three-phase greedy allocator with a two-pass linear scan over live intervals.

## Background: The CCC Allocator

CCC's original allocator processes live intervals in three greedy phases:

1. **Phase 1** — Sort all call-spanning values by priority. Assign callee-saved registers in order until they're exhausted.
2. **Phase 2** — Sort non-call-spanning values. Assign caller-saved registers.
3. **Phase 3** — Assign any remaining callee-saved registers to overflow non-call-spanning values.

The prioritization is loop-weighted: uses inside a loop at depth `d` contribute `10^d` to the score. This is good, but the greedy approach has a fundamental flaw: **it can't evict a lower-priority value to make room for a higher-priority one**. Once a register is assigned, it stays assigned.

Additionally, Phase 1 only considers call-spanning values. Non-call-spanning values in tight loops had to compete in Phase 2 for the small caller-saved pool (typically 2–4 registers on x86-64), often missing out entirely.

## Linear Scan Overview

Linear scan (Poletto & Sarkar, 1999) processes intervals in order of their start point. For each new interval:

1. **Expire** all active intervals that ended before this one starts. Free their registers.
2. **Find a free register.** If one exists, assign it.
3. **If no register is free**, compare the incoming interval's spill weight against the active interval with the lowest spill weight. Evict whichever is less important.

This makes a single linear pass with O(n log n) complexity — fast enough to run on every function.

## Data Structures

### `LiveRange` (`live_range.rs`)

```rust
pub struct LiveRange {
    pub value_id:    u32,
    pub start:       u32,          // program point of definition
    pub end:         u32,          // last use
    pub uses:        Vec<u32>,     // individual use points
    pub loop_depth:  u32,          // nesting depth (0 = no loop)
    pub priority:    u64,          // uses.len() * 10^loop_depth
    pub reg_hint:    Option<PhysReg>,  // preferred reg (from Copy source)
    pub spill_weight: f64,         // priority / range_length
}
```

Spill weight combines priority and interval length. A short, hot interval costs more to spill than a long, cold one.

### `LinearScanAllocator`

```rust
pub struct LinearScanAllocator {
    pub ranges:          Vec<LiveRange>,         // sorted by start
    pub active:          Vec<ActiveInterval>,    // currently live
    pub assignments:     FxHashMap<u32, PhysReg>,
    pub reg_free_until:  FxHashMap<PhysReg, u32>,
    pub spill_slots:     FxHashMap<u32, i32>,
    pub available_regs:  Vec<PhysReg>,
}
```

## The Allocation Algorithm

```
run():
  init all registers as free (reg_free_until = 0)
  for each range in ranges (sorted by start):
    allocate_range(range)

allocate_range(range):
  expire_old_intervals(range.start)    // free registers of ended intervals

  if free_register = find_free_register(range):
    assign free_register to range
    add range to active set

  else if spill_candidate = find_spill_candidate():
    // Only evict if the candidate has LOWER spill weight than the incoming range.
    // This ensures we always keep the most important value in the register.
    if candidate.spill_weight < range.spill_weight:
      remove candidate from assignments (it goes to stack)
      assign candidate's freed register to range
    else:
      spill incoming range (it goes to stack)

  else:
    spill incoming range
```

### Eviction Correctness

A subtle but critical fix from the original implementation: **the eviction comparison must consider the incoming interval's weight**, not just the active set. The old code always evicted the lowest-weight active interval regardless. This produced wrong results when the incoming interval was *less* important than everything already in registers — the test case that exposed this:

```rust
// Two overlapping intervals, one register
// Value 1: priority=3, spill_weight=0.03  (more important)
// Value 2: priority=2, spill_weight=0.02  (less important, arrives second)
//
// Correct:  value 1 keeps the register, value 2 spills
// Old code: always evicts value 1 (the only active), gives reg to value 2 — WRONG
```

The fix: if `active_weight >= incoming_weight`, spill the incoming range instead.

## Eligibility Filter

The allocator only processes values that can safely live in a general-purpose register. The filter excludes:

| Category | Why excluded |
|----------|--------------|
| `float`, `f64`, `long double` | Use XMM/x87 registers, not GPRs |
| `i128`, `u128` | Require register pairs |
| `i64`/`u64` on i686 | 32-bit target, need eax:edx pair |
| Alloca addresses | Must have stable stack addresses |
| Atomic pointers | `lock` prefix requires memory operand |
| `memcpy` dst/src pointers | Codegen uses `resolve_slot_addr` directly |
| VA arg pointers | Same — not register-aware |
| `CallIndirect` func pointers | Must dereference from memory |

Copy chains are propagated: if `%a = copy %b` and `%b` is ineligible, `%a` is also excluded.

## Two-Pass Design

LCCC runs the linear scan twice with different register pools:

```
Phase 1: LinearScanAllocator(eligible_intervals, callee_saved_regs)
  → callee-saved registers are safe for ALL values, including call-spanning ones
  → the allocator never needs to know whether an interval spans a call

Phase 2: LinearScanAllocator(unallocated_non_call_spanning, caller_saved_regs)
  → caller-saved regs can ONLY hold values that don't cross function calls
  → filter: !spans_any_call(interval, call_points)
```

This separation preserves ABI correctness: caller-saved registers are freely clobbered by callees, so any value assigned to one must not be live across a call instruction.

## Register Hints (Coalescing)

When a value is produced by a `Copy` instruction, the source and destination have the same value. If the source is allocated to register `R`, the destination is hinted to prefer `R`. If `R` is free when the destination is processed, the allocator assigns it — eliminating a register-to-register move.

The hint infrastructure exists in `live_range.rs` (`find_register_hints`, `reg_hint` field); full coalescing is a Phase 3 improvement.

## Performance Impact

| Benchmark | Δ vs CCC | Root cause |
|-----------|----------|------------|
| `arith_loop` (32 vars, 10M iters) | **+20% faster** | More callee-saved regs allocated; fewer load/store pairs in hot loop |
| `sieve` (primes to 10M) | **+25% faster** | Inner loop variables (counters, boundary) now in registers |
| `fib(40)` recursive | ≈ equal | Call-dominated; allocator has minimal effect |
| `matmul` 256×256 double | ≈ equal | FP throughput and GCC's AVX2 vectorization dominate |
| `qsort` 1M | ≈ equal | `libc qsort` call dominates — no LCCC code in hot path |
