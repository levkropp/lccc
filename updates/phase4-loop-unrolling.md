---
layout: doc
title: "Phase 4: Loop Unrolling"
description: "LCCC Phase 4 adds loop unrolling with early-exit intermediate checks, handling non-multiple trip counts without a cleanup loop."
date: 2026-03-19
prev_page:
  title: Phase 3 — TCE and Phi-Copy Coalescing
  url: /updates/phase3-tce-and-phi-coalescing
next_page:
  title: Benchmarks
  url: /docs/benchmarks
---

# Phase 4: Loop Unrolling
{:.doc-subtitle}
*2026-03-19*

Phase 4 adds loop unrolling to LCCC's optimization pipeline. Small inner loops are replicated 2×–8× per compiled cycle, with an exit check inserted between each copy. This handles any trip count correctly — no separate cleanup loop required.

---

## The problem

For a simple counting loop:

```c
for (int i = 0; i < N; i++)
    sieve[i] = 0;
```

The compiler emits one iteration per cycle:

```
header:
  %i = phi [0, %i_next]
  %cond = cmp %i < N
  branch %cond → body / exit

body:
  %ptr = gep sieve, %i
  store 0, %ptr
  branch latch

latch:
  %i_next = add %i, 1
  branch header
```

Every iteration pays the cost of the branch back to the header and the condition check. For a loop with a small body, that overhead is significant relative to the work done. GCC unrolls aggressively; CCC doesn't unroll at all.

---

## The approach: unroll with intermediate exit checks

Rather than unrolling into a pre-loop + main loop + cleanup structure (three separate loops), LCCC uses a single-loop strategy with exit checks between each copy:

```
header:
  %i = phi [0, %i_next]
  %cond = cmp %i < N
  branch %cond → body / exit          ← original check, unchanged

[original body]  →  exit_check_1

exit_check_1:
  %i_1 = add %i, 1
  %cond_1 = cmp %i_1 < N
  branch %cond_1 → exit / body_copy_2  ← fires if N = 1 mod 4

[body_copy_2 — %i renamed to %i_1]  →  exit_check_2

exit_check_2:
  %i_2 = add %i_1, 1
  %cond_2 = cmp %i_2 < N
  branch %cond_2 → exit / body_copy_3  ← fires if N = 2 mod 4

[body_copy_3 — %i renamed to %i_2]  →  exit_check_3

exit_check_3:
  %i_3 = add %i_2, 1
  %cond_3 = cmp %i_3 < N
  branch %cond_3 → exit / body_copy_4  ← fires if N = 3 mod 4

[body_copy_4 — %i renamed to %i_3]  →  latch

latch:
  %i_next = add %i_3, 1               ← was: add %i, 1
  branch header
```

**Why this works for any trip count:** whichever intermediate exit check fires first handles the partial cycle. If `N = 17` and `K = 4`, the loop runs four full cycles of 4 (iterations 0–15), then the header fires one more time (iteration 16), then exits via the original header check — no remainder loop needed.

**Header phi unchanged:** the phi still has exactly two incoming edges (preheader and latch). Only the value flowing in from the latch changes — from `%i + 1×step` to `%i + K×step`.

---

## Eligibility

The pass checks eight conditions before unrolling:

1. **Body size:** loop body (excluding header and latch) has ≤8 blocks.
2. **Single latch:** exactly one back-edge to the header; latch terminates with an unconditional branch.
3. **Preheader:** a unique predecessor outside the loop (required for correct phi analysis).
4. **No nested loops:** no body-work block is a header of another loop — unrolling an outer loop that contains an inner loop would break the inner loop's structure.
5. **No side-effectful ops:** no `call`, `atomicrmw`, `cmpxchg`, `atomic_load`, `atomic_store`, `dynalloca`, or `inline_asm` in the body — these can't be duplicated safely.
6. **Basic IV:** a phi in the header whose back-edge value is `add(%iv, const_step)` in the latch.
7. **Detectable exit:** the header's `CondBranch` condition traces through at most one cast to a `Cmp` that uses `%iv` on one side and a loop-invariant value on the other.
8. **Exit-block phi safety:** any phi in the exit block that receives a value from the loop must receive a loop-invariant value — so each new exit edge can carry the same value.

---

## Unroll factors

```rust
fn choose_unroll_factor(body_inst_count: usize) -> u32 {
    match body_inst_count {
        0..=8   => 8,
        9..=20  => 4,
        21..=60 => 2,
        _       => 1,   // too large — skip
    }
}
```

A loop body with 1–8 instructions gets unrolled 8×. This is the common case for inner loops like sieve marking or array initialization.

---

## SSA validity: renaming definitions, not just uses

When cloning body blocks, every SSA value defined in the clone must get a fresh ID. This requires renaming both **uses** (operands that reference `%iv` or other loop values) and **definitions** (the `dest` field of each instruction). Missing the definition rename produces duplicate value IDs — invalid SSA — and would silently corrupt later optimization passes.

The fix is `rename_inst_dest`, applied to every cloned instruction after operand renaming:

```rust
let new_insts: Vec<Instruction> = orig.instructions.iter()
    .map(|inst| {
        let mut cloned = inst.clone();
        replace_values_in_inst(&mut cloned, vmap);  // rename uses
        rename_inst_dest(&mut cloned, vmap);          // rename defs
        cloned
    })
    .collect();
```

---

## Results

Best-of-7 wall-clock, Linux x86-64, GCC 15.2.0 `-O2`:

| Benchmark | LCCC | CCC | GCC | vs CCC | vs GCC |
|-----------|-----:|----:|----:|:------:|:------:|
| `arith_loop` | **0.103 s** | 0.146 s | 0.068 s | +42% | 1.50× |
| `sieve` | **0.036 s** | 0.045 s | 0.024 s | +25% | 1.50× |
| `qsort` | 0.096 s | 0.095 s | 0.087 s | ≈equal | 1.10× |
| `fib(40)` | 0.352 s | 0.354 s | 0.096 s | ≈equal | 3.68× |

The sieve prime-counting loop (`for (i=2; i<=N; i++) if (sieve[i]) count++`) is unrolled 8× — the inner sieve-marking loop (`j += i`) has a variable step and is not unrolled. The counting loop is not the bottleneck, so the wall-clock improvement is modest.

`arith_loop` has 291 body instructions, far above the 60-instruction cutoff — it is not unrolled. Its improvement over CCC comes entirely from the earlier phases (linear scan + phi-copy coalescing).

All 514 unit tests pass. The new pass has 6 dedicated unit tests covering basic unrolling, call inhibition, large-body inhibition, missing-preheader inhibition, nested loop handling, and SSA uniqueness.

---

## What's next

The matmul benchmark (`0.020 s` LCCC vs `0.004 s` GCC) still shows a 4.86× gap — GCC uses AVX2 packed double operations. Phase 5 targets auto-vectorization.
