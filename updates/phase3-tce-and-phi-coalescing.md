---
layout: doc
title: "Phase 3: Tail-Call Elimination and Phi-Copy Coalescing"
description: "LCCC Phase 3 delivers 139× speedup on tail-recursive code and +19% additional gain on register-pressure loops via two complementary backend improvements."
date: 2026-03-19
prev_page:
  title: Roadmap
  url: /docs/roadmap
next_page:
  title: Benchmarks
  url: /docs/benchmarks
---

# Phase 3: Tail-Call Elimination and Phi-Copy Coalescing
{:.doc-subtitle}
*2026-03-19*

Phase 3 ships two independent optimizations. Together they bring LCCC to **+41% faster than CCC** on register-pressure code and **139× faster** on tail-recursive code.

---

## Phase 3a — Tail-Call Elimination

### The problem

`sum(10000000, 0)` takes 1.09 seconds in CCC. LCCC runs it in 8 ms. The difference is 10 million stack frames.

```c
static long sum(int n, long acc) {
    if (n <= 0) return acc;
    return sum(n - 1, acc + n);   // ← tail call: result returned immediately
}
```

A tail call is a recursive call whose result flows directly to a `return` — no further computation. The recursive call could be a loop back-edge, but CCC's backend doesn't know that.

### The fix

TCE (Tail-Call Elimination) runs as an IR-level pass after inlining, before the main optimization loop. It:

1. Detects self-recursive calls where the result feeds directly into a `Return`
2. Inserts a loop header block with one Phi node per function parameter
3. Renames all parameter references in the function body to the Phi outputs
4. Replaces `call sum(n-1, acc+n); return result` with assignments to the Phi inputs + an unconditional branch to the loop header

The result is a proper SSA loop that LICM, IVSR, and GVN can optimize in the normal pipeline.

```
Before TCE:                          After TCE:
                                      loop_header:
entry:                                  %n_cur   = Phi [%n from entry, %n_next from body]
  ...                                   %acc_cur = Phi [%acc from entry, %acc_next from body]
body:                                 body:
  %result = Call sum(%n-1, %acc+%n)    %n_next   = BinOp sub %n_cur, 1
  Return %result                        %acc_next = BinOp add %acc_cur, %n_cur
                                        Branch → loop_header
```

After LICM and IVSR, the resulting loop compiles to 3 instructions — identical to GCC's output.

### Results

| | LCCC | CCC | GCC |
|---|---|---|---|
| `sum(10000000, 0)` | **8 ms** | 1090 ms | 8 ms |
| vs CCC | **139× faster** | baseline | 139× faster |

The pass lives in [`src/passes/tail_call_elim.rs`](https://github.com/levkropp/lccc/blob/master/src/passes/tail_call_elim.rs) (747 lines, 8 unit tests). Disable with `CCC_DISABLE_PASSES=tce`.

---

## Phase 3b — Phi-Copy Stack Slot Coalescing

### The problem

After Phase 2 (linear scan), `arith_loop` was 20% faster than CCC. But 20 redundant memory copies were still happening every loop iteration.

Here's why. CCC's phi elimination converts each SSA phi node to `Copy` instructions in predecessor blocks:

```
SSA form:
  loop_header:
    %b = Phi [2 from entry, %b_next from backedge]

After phi elimination:
  entry:    %b = Copy 2         ← initializes b's slot
  backedge: %b = Copy %b_next  ← updates b's slot for next iteration
```

Now `%b` has *two definitions* (entry + backedge). The existing copy coalescing marks multi-defined values as ineligible, so `%b` and `%b_next` get separate stack slots. The backedge copy becomes:

```asm
movq -280(%rbp), %rax    ; load %b_next from its slot
movq %rax, -104(%rbp)    ; store to %b's slot
```

With 32 loop variables, that's 22 pairs of redundant `movq mem, %rax; movq %rax, mem` — every iteration, 10 million times.

### The fix

The existing coalescing only merged copies in the *forward* direction: alias `dest → src` (dest borrows src's slot). This is safe when both are in the same block. For phi copies, both conditions fail: dest is multi-defined, and dest's uses span different blocks.

The key insight: **reverse the alias direction**. For the phi-copy pattern, alias `src → dest` instead — `%b_next` borrows `%b`'s slot. This is safe because:

- `%b_next` has exactly one use: the Copy. After aliasing, that Copy is a same-slot no-op.
- `%b`'s slot is already live across all of `%b`'s uses (including the loop header), so `%b_next` writing to it in the backedge block is fine.
- `%b` being multi-defined just means the slot gets written in multiple places — correct by design.

The code change in `src/backend/stack_layout/copy_coalescing.rs`:

```rust
// Before: rejected because d (phi dest) is multi-defined
if multi_def_values.contains(&d) || multi_def_values.contains(&s) {
    continue;
}

// After: d being multi-defined is OK when it's the slot owner (root)
// Only reject if s (the aliased value) is multi-defined
if multi_def_values.contains(&s) { continue; }
// ... detect phi-copy pattern ...
if src_in_copy_block && dest_cross_block {
    raw_aliases.push((s, d));  // reversed: s borrows d's wider-live slot
    continue;
}
// Standard coalescing: still requires d to be single-defined
if multi_def_values.contains(&d) { continue; }
```

The code generator already had a same-slot no-op check in `generate_copy`:

```rust
if ds.0 == ss.0 {
    // same slot — update the accumulator cache and return
    return;
}
```

So the "redundant copy" is simply never emitted.

### Results

| | Before (Phase 2 only) | After (Phase 2 + 3b) | CCC |
|---|---|---|---|
| `arith_loop` | 0.124s | **0.104s** | 0.147s |
| vs CCC | +20% faster | **+41% faster** | baseline |
| Assembly lines | 550 | 507 | ~550 |

The `arith_loop` loop body loses ~43 assembly lines — the 22 redundant `movq` pairs, now gone.

---

## Combined results

| Benchmark | LCCC | CCC | GCC -O2 | LCCC vs CCC |
|-----------|-----:|----:|--------:|:-----------:|
| `arith_loop` | **0.104s** | 0.147s | 0.068s | **+41%** |
| `sieve` | **0.037s** | 0.044s | 0.024s | **+19%** |
| `qsort` | 0.098s | 0.096s | 0.087s | ≈ equal |
| `fib(40)` | 0.352s | 0.355s | 0.096s | ≈ equal |
| `matmul` | 0.027s | 0.029s | 0.003s | ≈ equal |
| `tce_sum` | **0.008s** | 1.09s | 0.008s | **139×** |

508 unit tests pass. All outputs are byte-identical to GCC's.

---

## What's next

Phase 4 is loop unrolling. The `arith_loop` inner loop body is still ~350 instructions; unrolling 4× would reduce the branch overhead and expose more ILP to the CPU's out-of-order engine. Target: close the arith_loop gap from 1.53× to ~1.2× vs GCC.

See the [Roadmap](/lccc/docs/roadmap) for the full plan.
