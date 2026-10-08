---
layout: doc
title: "Phase 5: Floating-Point Peephole Optimization"
description: "LCCC Phase 5 eliminates GPR↔XMM round-trips, folds memory operands into FP instructions, and removes stack spills — cutting matmul time by 41% (6.0× → 4.0× vs GCC)."
date: 2026-03-19
prev_page:
  title: Phase 4 — Loop Unrolling
  url: /updates/phase4-loop-unrolling
next_page:
  title: Benchmarks
  url: /docs/benchmarks
---

# Phase 5: Floating-Point Peephole Optimization
{:.doc-subtitle}
*2026-03-19*

Phase 5 adds seven new peephole patterns that target floating-point code. Together they reduce the 256×256 matmul inner loop from 33 instructions to 20 and cut wall-clock time from **0.027 s → 0.016 s** — a **41% speedup** that brings the GCC gap from 6.0× to **4.0×**.

---

## The problem

CCC's code generator treats every value as a 64-bit integer. Floating-point operands are loaded into GPRs (`%rax`, `%rcx`), spilled to stack slots, then transferred to XMM registers for the actual FP operation, and transferred back to GPRs for storage. A single `C[i][j] += A[i][k] * B[k][j]` compiles to roughly this:

```asm
# Load A[i][k]  (3 instructions — should be 1)
movq %r10, %rcx            # copy pointer to rcx
movq (%rcx), %rax           # deref through rcx into GPR
movq %rax, -32(%rbp)        # spill to stack

# Load B[k][j]  (3 instructions — should be 1)
movq %r13, %rcx
movq (%rcx), %rax
movq %rax, -40(%rbp)

# Compute A*B   (7 instructions — should be 2)
movq -32(%rbp), %rax        # reload A from stack
movq %rax, %xmm0            # GPR → XMM
movq -40(%rbp), %rcx        # reload B from stack
movq %rcx, %xmm1            # GPR → XMM
mulsd %xmm1, %xmm0          # actual multiply
movq %xmm0, %rax            # XMM → GPR
movq %rax, -48(%rbp)         # spill product

# Load C[i][j], accumulate, store  (8 instructions — should be 3)
movq (%rcx), %rax
movq %rax, %xmm0
addsd -48(%rbp), %xmm0
movq %xmm0, %rax
movq %rax, %rdx
movq %r15, %rcx
movq %rdx, (%rcx)
...

# Loop overhead: 12 instructions for j++, pointer advance
```

**33 instructions** per iteration. GCC's vectorized inner loop uses 7.

---

## The fix: seven peephole patterns

All patterns live in `src/backend/x86/codegen/peephole/passes/local_patterns.rs` and `memory_fold.rs`, added to the Phase 1 local optimization loop and the Phase 3/4b cleanup rounds. Each pattern fires on assembly text after instruction classification and works with the existing `LineKind`/`LineStore` infrastructure.

### Pattern A — XMM load round-trip elimination

```
LoadRbp{rax, -32, Q}  +  "movq %rax, %xmm0"
────────────────────────────────────────────────
→   "movsd -32(%rbp), %xmm0"                    (1 instruction, was 2)
```

Replaces `movq -32(%rbp), %rax; movq %rax, %xmm0` with a direct FP load. The GPR intermediary is eliminated.

### Pattern B — XMM store round-trip elimination

```
"movq %xmm0, %rax"  +  StoreRbp{rax, -48, Q}
────────────────────────────────────────────────
→   "movsd %xmm0, -48(%rbp)"                    (1 instruction, was 2)
```

With a **liveness check**: `%rax` must not be read after the store. The check handles `movq %<non-rax>, %rax` as a pure write (not a read), avoiding a false-positive that caused the first correctness bug.

### FP memory fold

```
"movsd -40(%rbp), %xmm1"  +  "mulsd %xmm1, %xmm0"
────────────────────────────────────────────────────
→   "mulsd -40(%rbp), %xmm0"                    (1 instruction, was 2)
```

Folds a stack-slot load into the subsequent FP operation as a memory source operand. Handles `mulsd`, `addsd`, `subsd`, `divsd`.

### Pattern D — pointer-to-XMM load folding

```
"movq (%rcx), %rax"  +  "movq %rax, %xmm0"
────────────────────────────────────────────
→   "movsd (%rcx), %xmm0"                       (1 instruction, was 2)
```

After Pattern E (below) removes a dead intervening store, Pattern D folds the remaining pair.

### Pattern E — dead stack store before XMM use

```
StoreRbp{rax, -40, Q}  +  "movq %rax, %xmm0"   (where -40(%rbp) is dead)
───────────────────────────────────────────────────
→   (NOP)  +  "movq %rax, %xmm0"               (store eliminated)
```

Scans forward up to 32 instructions to verify the stack slot is never read before being overwritten or reaching a control-flow boundary. Creates the adjacency for Pattern D to fire.

### Pattern F — 4-instruction store-through-pointer folding

```
"movq %xmm0, %rax"
"movq %rax, %rdx"
"movq %r15, %rcx"
"movq %rdx, (%rcx)"
──────────────────────
→   "movsd %xmm0, (%r15)"                       (1 instruction, was 4)
```

Detects the accumulator-model chain where an XMM value is shuffled through two GPRs before being stored to a pointer address. Eliminates 3 instructions.

### Pattern G — rcx address-register copy elimination

```
"movq %r10, %rcx"  +  "movq (%rcx), %rax"
──────────────────────────────────────────
→   "movq (%r10), %rax"                         (1 instruction, was 2)
```

CCC always copies the address to `%rcx` before dereferencing. This pattern folds the copy into the load when `%rcx` is dead after the dereference. Also handles the `movsd` variant for direct FP loads.

### Pattern H — pointer-deref stack elimination

```
"movq (%r10), %rax"  +  StoreRbp{rax, -32, Q}  +  [gap]  +  "movsd -32(%rbp), %xmm0"
────────────────────────────────────────────────────────────────────────────────────────
→   (NOP)  +  (NOP)  +  [gap]  +  "movsd (%r10), %xmm0"   (1 instruction, was 3)
```

The most impactful pattern. Scans forward from a GPR store to find the eventual FP use of the same stack slot. Replaces the stack-slot reference with the original pointer register, eliminating the GPR load and stack spill entirely. Requires: pointer register unchanged across the gap, stack slot dead after the FP use, `%rax` overwritten before being read in the gap.

### Pattern I — FP spill elimination around load

```
"movsd %xmm0, -48(%rbp)"    # spill product
...                           # address calculation (no xmm usage)
"movsd (%r15), %xmm0"        # load C[i][j] (overwrites xmm0)
"addsd -48(%rbp), %xmm0"     # reload product
──────────────────────────────────────────────────────────────
→   (NOP)                    # spill eliminated
    ...
    "movsd (%r15), %xmm1"   # C loaded into xmm1 instead
    "addsd %xmm1, %xmm0"    # register-only add
```

Detects when `%xmm0` is spilled to the stack, then overwritten by a load, then the spill is read back. Redirects the intervening load to `%xmm1`, keeping the original value in `%xmm0` and turning the stack reload into a register-register `addsd`.

---

## Result: the matmul inner loop

After all nine patterns fire, the inner loop compiles to:

```asm
.LBB10:
    movq %rbx, %r14              # j index
    shlq $3, %r14                # j * 8 (byte offset)
    movsd (%r10), %xmm0          # A[i][k]           ← Pattern H
    mulsd (%r13), %xmm0          # × B[k][j]          ← Pattern H
    movq %r8, %rcx               # (dead — not yet eliminated)
    movq %r14, %rax              # address calculation
    addq %r8, %rax               #   &C[i][j]
    movq %rax, %r15              #
    movsd (%r15), %xmm1          # C[i][j]            ← Patterns D+G+I
    addsd %xmm1, %xmm0           # C + A×B             ← Pattern I
    movsd %xmm0, (%r15)          # store               ← Pattern F
    movq %r12, %r14              # loop counter
    addq $1, %r14                #   j++
    movslq %r14d, %rax           #
    movq %rax, %r15              #
    movq %r13, %rax              # advance B pointer
    leaq 8(%rax), %rax           #   B += 8
    movq %rax, %r14              #
    movq %r15, %r12              #
    movq %rax, %r13              #
    jmp .LBB9                    # back to loop header
```

**20 instructions** (was 33). The FP core is 5 clean instructions with direct memory operands — essentially matching GCC's scalar code structure.

For comparison, GCC `-O2` generates:

```asm
.L3:
    movapd (%rcx,%rax), %xmm0    # load 2 doubles of B
    mulpd  %xmm1, %xmm0          # × broadcast(A[i][k])
    addpd  (%rdx,%rax), %xmm0    # + 2 doubles of C
    movaps %xmm0, (%rdx,%rax)    # store
    addq   $16, %rax
    cmpq   $2048, %rax
    jne    .L3
```

**7 instructions** processing **2 doubles per iteration** — the remaining 4× gap comes from (1) SSE2 vectorization (2× throughput) and (2) tighter loop control (byte-offset counter vs integer counter with register shuffling).

---

## Benchmark results

Best-of-10 wall-clock, Linux x86-64:

| Benchmark | Before Phase 5 | After Phase 5 | Change | vs GCC -O2 |
|-----------|---------------:|:-------------:|:------:|:----------:|
| **matmul 256²** | 0.027 s | **0.016 s** | **-41%** | 4.0× (was 6.0×) |
| arith_loop | 0.103 s | 0.103 s | — | 1.52× |
| sieve | 0.036 s | 0.037 s | — | 1.48× |
| fib(40) | 0.353 s | 0.353 s | — | 3.69× |
| qsort 1M | 0.096 s | 0.098 s | — | 1.12× |

No regressions on any benchmark. All outputs remain byte-identical to GCC.

---

## What's still needed

The remaining 4× matmul gap vs GCC breaks down as:

**~2× from no SIMD vectorization.** GCC auto-vectorizes the j-loop with `mulpd`/`addpd`, processing 2 doubles per SSE2 iteration (4× with AVX). LCCC operates purely scalar. Fixing this requires either:
- An IR-level auto-vectorizer that recognizes the `C[i][j] += A[i][k] * B[k][j]` pattern and emits packed operations (the `FmaF64x2` intrinsic is already defined in `src/ir/intrinsics.rs` and lowered in `src/backend/x86/codegen/intrinsics.rs`), or
- A peephole-level "SLP vectorizer" that packs adjacent scalar `movsd`/`mulsd`/`addsd` instructions into their packed equivalents

**~1.5× from loop overhead.** GCC uses a single byte-offset counter (`addq $16, %rax; cmpq $2048, %rax`) — 3 loop-control instructions. LCCC uses 10 instructions to increment the integer counter, sign-extend, advance the B pointer, and shuffle registers. Reducing this requires:
- Loop strength reduction (replace `j*8 + base` recomputation with a single incrementing pointer)
- Copy propagation across loop back-edges to eliminate the register shuffle
- Dead-code elimination for the orphaned `movq %r8, %rcx` (1 instruction, left behind after Pattern G eliminated its consumer)

**~1.3× from A[i][k] invariant hoisting.** A[i][k] is constant across the j-loop but reloaded every iteration (`movsd (%r10), %xmm0`). GCC hoists this to the outer k-loop and broadcasts it. This requires loop-invariant code motion (LICM) operating on the lowered machine code, or an earlier IR-level LICM that can see through the pointer dereference.

---

## Files changed

| File | Change |
|------|--------|
| `src/backend/x86/codegen/peephole/passes/local_patterns.rs` | Patterns A–I, helpers `rax_is_live_at`, `rcx_is_live_at`, `rbp_offset_dead_after` |
| `src/backend/x86/codegen/peephole/passes/memory_fold.rs` | `fold_fp_memory_operands` (FP memory fold) |
| `src/backend/x86/codegen/peephole/passes/mod.rs` | Wired all new passes into Phase 1, 3, and 4b |
| `src/ir/intrinsics.rs` | `FmaF64x2` variant (infrastructure for future vectorization) |
| `src/backend/x86/codegen/intrinsics.rs` | `FmaF64x2` emission (SSE2 packed multiply-add) |
| `src/backend/i686/codegen/intrinsics.rs` | `FmaF64x2` placeholder |
| `src/backend/arm/codegen/intrinsics.rs` | `FmaF64x2` in x86-only arm |
| `src/backend/riscv/codegen/intrinsics.rs` | `FmaF64x2` in x86-only arm |
