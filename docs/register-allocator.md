---
layout: doc
title: Register Allocator
description: Deep dive into LCCC's linear scan register allocator — pools, phi coalescing, and the loop-value steal.
prev_page:
  title: Architecture
  url: /docs/architecture
next_page:
  title: Optimization Passes
  url: /docs/optimization-passes
---

# Register Allocator
{:.doc-subtitle}
LCCC replaces CCC's three-phase greedy allocator with a linear scan over live intervals, plus a conflict-safe register steal for hot loop-carried values.

## Background: The CCC Allocator

CCC's original allocator processes live intervals in three greedy phases:

1. **Phase 1** — Sort all call-spanning values by priority. Assign callee-saved registers in order until they're exhausted.
2. **Phase 2** — Sort non-call-spanning values. Assign caller-saved registers.
3. **Phase 3** — Assign any remaining callee-saved registers to overflow non-call-spanning values.

The prioritization is loop-weighted: uses inside a loop at depth `d` contribute `10^d` to the score. This is good, but the greedy approach has a fundamental flaw: **it can't evict a lower-priority value to make room for a higher-priority one**. Once a register is assigned, it stays assigned.

Additionally, Phase 1 only considers call-spanning values. Non-call-spanning values in tight loops had to compete in Phase 2 for the small caller-saved pool, often missing out entirely.

## Linear Scan Overview

Linear scan (Poletto & Sarkar, 1999) processes intervals in order of their start point. For each new interval:

1. **Expire** all active intervals that ended before this one starts. Free their registers.
2. **Find a free register.** If one exists, assign it.
3. **If no register is free**, the incoming range spills.

A note on step 3: the classic algorithm *evicts* the active interval with the lowest spill weight when the incoming range is hotter. LCCC's allocator does **not** evict — whole-interval eviction was measured to miscompile in this codebase (the interaction between evicted ranges and stack-slot assignment), so the scan keeps first-come assignments. The hot-vs-cold rebalance that eviction was meant to provide is instead done *after* the scan by the register steal (below), which is safe by construction. An IR-level loop-transparent splitting variant exists behind `CCC_LOOP_SPLIT` and is documented as a measured dead-end: in-loop values can't be transparently split.

## The Register Pools

```
Phase 1: LinearScanAllocator(eligible_intervals, callee_saved_regs)
  → callee-saved registers are safe for ALL values, including call-spanning ones

Phase 2: LinearScanAllocator(unallocated_non_call_spanning, caller_saved_regs)
  → caller-saved regs can ONLY hold values that don't cross function calls

Phase 3: LinearScanAllocator(F64/vector intervals, xmm_regs)   [AArch64, x86-64]
  → F64 accumulators and vector values get FP/SIMD registers
    (AArch64: d16–d23; d24–d31 reserved for promoted loop accumulators)
```

Use counts are loop-depth weighted (`10^depth` per use) and count pointer/base uses
(`Store.ptr`, `Load.ptr`, `GEP.base`) as well as operand uses, so carried pointers rank by
their true temperature.

## Post-Scan Register Steal (AArch64)

The scan processes ranges in start order, so cold function-spanning values (array bases,
globals, prologue pointers) win every callee-saved register simply by starting first —
leaving hot inner-loop-carried phi values (IVs, accumulators, carried pointers) stack-homed.
fannkuch's flip loop was the canonical victim.

For each hot loop-carried phi dest the scan **missed**, the steal:

1. Computes, for every callee-saved register, the set of holders whose live intervals
   overlap the hot value (all interval segments, not just the first).
2. Picks the register whose conflicting holders have the coldest total loop-weighted use
   count — strictly colder than the hot value, or no steal happens.
3. **Fully deallocates those holders** back to the stack (indistinguishable from never
   having been assigned — the default stack path handles all their uses), and assigns the
   register to the hot value.

This is safe where in-scan eviction was not: an evicted holder loses its register for its
*entire* interval, no live range is ever split, and remaining holders are provably
non-overlapping. When the scan already housed every hot value, the steal is a no-op — so
functions whose loops are already well-served (e.g. spectral_norm's out-of-line kernels) are
untouched. Registers holding phi-coalesce participants are left alone to keep backedge
coalescing intact.

Controls: `CCC_NO_LOOP_PIN` disables; `CCC_LOOP_PIN=N` caps steals per function (default 2);
`CCC_DEBUG_LOOP_PIN` traces decisions.

## Phi Coalescing (Registers and Stack Slots)

Loop-carried phi variables lower to a pair of copies: an initializer and a backedge
`%old = copy %new`. Coalescing the two values onto the same storage makes that copy a no-op.

- **Register coalescing** (`regalloc.rs`): the backedge source inherits the phi dest's
  register, after a conflict check against all other values assigned that register. The
  backedge `mov` disappears.
- **Slot coalescing** (`stack_layout/copy_coalescing.rs`): when both values are stack-homed,
  the backedge source borrows the phi dest's stack slot — including the common
  constant-initializer case, which earlier copy coalescing never handled. Safety comes from
  the same detector proof (the phi dest is dead after the backedge source is defined);
  certified pairs bypass the generic def/last-use interference check, which would otherwise
  reject them because the phi dest is live again on later iterations. The `ldr`+`str`
  per-iteration shuffle disappears — 22 instructions/iteration in arith_loop.

## Eligibility Filter

The allocator only processes values that can safely live in a general-purpose register. The filter excludes:

| Category | Why excluded |
|----------|--------------|
| `float`, `f64`, `long double` | Use FP/SIMD registers, not GPRs |
| `i128`, `u128` | Require register pairs |
| `i64`/`u64` on i686 | 32-bit target, need eax:edx pair |
| Alloca addresses | Must have stable stack addresses |
| Atomic pointers | `lock` prefix / atomics require memory operands |
| `memcpy` dst/src pointers | Codegen uses `resolve_slot_addr` directly |
| VA arg pointers | Same — not register-aware |
| `CallIndirect` func pointers | Must dereference from memory |

Copy chains are propagated: if `%a = copy %b` and `%b` is ineligible, `%a` is also excluded.

## Data Structures

### `LiveRange` (`live_range.rs`)

```rust
pub struct LiveRange {
    pub value_id:    u32,
    pub start:       u32,          // program point of definition
    pub end:         u32,          // last use
    pub uses:        Vec<u32>,     // individual use points
    pub loop_depth:  u32,          // nesting depth (0 = no loop)
    pub priority:    u64,          // loop-depth-weighted use count
    pub reg_hint:    Option<PhysReg>,  // preferred reg (from Copy source)
    pub spill_weight: f64,         // priority / range_length
}
```

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

## Performance Impact (AArch64 suite)

| Benchmark | Δ | Mechanism |
|-----------|---|-----------|
| `fannkuch` | 2.69× → **1.85×** vs GCC | Register steal houses flip-loop pointers/IVs |
| `arith_loop` | 2.09× → **1.15×** vs GCC | Loop-backedge slot coalescing |
| `spectral_norm` | 1.63× → **1.19×** vs GCC | Steal + F64 loop promotion |
| `matmul` | **1.17× faster** | NEON vector ops + FP register pools |
