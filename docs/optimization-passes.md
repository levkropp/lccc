---
layout: doc
title: Optimization Passes
description: The CCC/LCCC optimizer pass pipeline — what runs, in what order, and why.
prev_page:
  title: Register Allocator
  url: /docs/register-allocator
next_page:
  title: Benchmarks
  url: /docs/benchmarks
---

# Optimization Passes
{:.doc-subtitle}
LCCC inherits CCC's full optimizer. All optimization levels (`-O0` through `-O3`, `-Os`, `-Oz`) run the same pipeline — by design.

## Why One Pipeline?

Having separate optimization tiers creates distinct code paths through the optimizer. Bugs that only appear at `-O2` can hide for months if most testing uses `-O0`. CCC and LCCC prioritize correctness over partial speedups: by always running all passes, every test exercises the full optimizer, and divergences are caught early.

The `-O` flags still control the `__OPTIMIZE__` and `__OPTIMIZE_SIZE__` preprocessor macros, which some projects (like the Linux kernel) use for conditional compilation.

## Pass Order

The optimizer runs up to three full iterations. Each iteration runs this pipeline:

```
1.  CFG simplification      — remove dead blocks, thread jump chains, simplify branches
2.  Copy propagation        — replace uses of copies with original values
2a. Division-by-constant    — replace idiv/div with multiply-shift (iter 0 only)
2b. Integer narrowing       — shrink operation widths (e.g. i64 add → i32 add)
3.  Algebraic simplification — strength reduction, identity removal
4.  Constant folding        — evaluate constant expressions at compile time
5.  GVN + LICM + IVSR       — shared CFG analysis (computed once per function):
    ├── GVN                 — global value numbering / CSE across dominated blocks
    ├── LICM                — loop-invariant code motion
    └── IVSR                — induction variable strength reduction
7.  If-conversion           — convert branch+phi diamonds to cmov/Select
8.  Copy propagation        — clean up copies created by GVN, simplify, LICM
9.  Dead code elimination   — remove dead instructions (excluded from convergence check)
10. CFG simplification      — clean up after DCE empties blocks
10.5 IPCP                   — interprocedural constant propagation (all iterations)
```

**Phase 0** (before the loop):
- Tail-call elimination (TCE) — converts self-recursive tail calls to loops
- Function inlining
- mem2reg
- Initial constant-fold/copy-prop round
- **Vectorization** (iteration 0 only) — transforms reduction loops to AVX2/SSE2 SIMD

**Phase 11** (after the loop): dead static function elimination — removes `static inline` functions that became unreferenced after inlining.

## Convergence

Iterations stop when:
- No pass made any change (fixpoint reached), **or**
- This iteration made fewer than 1/20th the changes of the first iteration (diminishing returns).

DCE changes are excluded from the convergence check because DCE's large removal counts inflate the first-iteration baseline and cause premature exit.

## Pass Interactions

The passes form a dependency graph. LCCC uses a `should_run!` macro to skip passes when their upstream passes made no changes in the previous iteration:

| Pass | Only re-runs if... |
|------|-------------------|
| CFG simplify | constfold changed (constant branches) or DCE changed (empty blocks) |
| Copy prop | CFG, GVN, LICM, or if-convert changed |
| Simplify | Copy prop or narrowing changed |
| Constfold | Copy prop, narrowing, simplify, or if-convert changed |
| GVN | CFG, copy prop, or simplify changed |
| LICM | CFG, copy prop, or GVN changed |
| DCE | GVN, LICM, if-convert, or copy prop2 changed |

## Key Passes

### GVN (Global Value Numbering)

Eliminates redundant computations across basic blocks. If two blocks both compute `a + b` where `a` and `b` have the same value, GVN replaces the second with a copy of the first. Operates on `BinOp`, `UnaryOp`, `Cmp`, `Cast`, `GetElementPtr`, and `Load` instructions within the dominator tree.

GVN, LICM, and IVSR share a single `CfgAnalysis` (dominator tree + loop nesting) per function per iteration, saving significant compile time on large translation units.

### LICM (Loop-Invariant Code Motion)

Hoists computations out of loops when their operands don't change within the loop. Identifies natural loops via the dominator tree, then moves invariant instructions to loop preheaders. Critical for inner-loop performance on code like matrix multiply.

### IPCP (Interprocedural Constant Propagation)

When a function is always called with the same constant argument, IPCP specializes the function for that constant and folds the resulting dead branches. Important for Linux kernel code (`IS_ENABLED()`, `cpucap_is_possible()` chains) where `static inline` wrappers gate large blocks of dead code.

### Division-by-Constant

Replaces `idiv`/`div` instructions (20–90 cycle latency on modern CPUs) with multiply-and-shift sequences. Runs only on the first iteration, before narrowing and constant folding can further simplify the expanded sequence. Disabled on i686 where 64-bit multiply overflow semantics differ.

## Disabling Passes

For debugging, individual passes can be disabled:

```bash
CCC_DISABLE_PASSES="gvn,licm" ./target/release/ccc input.c -o output
CCC_DISABLE_PASSES="all"      ./target/release/ccc input.c -o output
```

Pass names: `cfg`, `copyprop`, `narrow`, `simplify`, `constfold`, `gvn`, `licm`, `ifconv`, `dce`, `ipcp`, `inline`, `ivsr`, `divconst`.

Timing data is available via:

```bash
CCC_TIME_PASSES=1 ./target/release/ccc input.c -o output 2>&1 | grep PASS
```

## LCCC-Specific Passes

LCCC adds two optimization passes that run before CCC's main optimizer loop.

### Tail-Call Elimination (`tce`)

Converts self-recursive tail calls to back-edge branches. A tail call is a recursive call whose result is returned immediately — `return f(args)` with no further computation.

```c
// Before: 10M stack frames
long sum(int n, long acc) {
    if (n <= 0) return acc;
    return sum(n - 1, acc + n);
}

// After TCE: tight counted loop (identical to GCC output)
long sum(int n, long acc) {
loop:
    if (n <= 0) return acc;
    acc += n; n -= 1; goto loop;
}
```

TCE runs once after inlining, before the main optimization loop, so that LICM, IVSR, and GVN can subsequently optimize the resulting loop.

**Pass name:** `tce` (disable with `CCC_DISABLE_PASSES=tce`)

**Implementation:** [`src/passes/tail_call_elim.rs`](https://github.com/levkropp/lccc/blob/master/src/passes/tail_call_elim.rs)

### Phi-Copy Stack Slot Coalescing (backend)

This is a backend optimization in `src/backend/stack_layout/copy_coalescing.rs`, not a pass in the traditional sense. It runs during stack layout, before code generation.

When CCC's phi elimination lowers SSA phi nodes to Copy instructions, it creates separate stack slots for the phi destination and its backedge update value. For a 32-variable loop, this generates ~20 redundant stack-to-stack `movq` pairs per iteration.

LCCC detects the phi-copy pattern — where the source is defined and killed in the backedge block — and aliases the source to use the phi destination's wider-live slot. The Copy becomes a same-slot no-op and is dropped by `generate_copy`.

**Result:** `arith_loop` (32 variables): 550 → 507 assembly lines; 0.124s → 0.104s.

## LCCC-Specific: Reduction Vectorization

**Added in Phase 8** — LCCC detects and transforms reduction loops into AVX2/SSE2 SIMD operations.

### What Gets Vectorized

Simple reduction patterns:
```c
// Sum reduction
double sum = 0.0;
for (int i = 0; i < n; i++)
    sum += arr[i];

// Dot product
double dot = 0.0;
for (int i = 0; i < n; i++)
    dot += a[i] * b[i];
```

### Transformation Strategy

1. **Pattern detection**: Identifies loops with a scalar accumulator PHI and a single reduction operation
2. **Loop splitting**: Divides loop bound by vector width (4 for AVX2, 2 for SSE2)
3. **Vector body**: Replaces scalar ops with vector intrinsics (VecLoad, VecAdd, VecMul)
4. **Horizontal reduction**: Extracts scalar from final vector (`vextractf128` + `vunpckhpd` + `vaddsd`)
5. **Remainder loop**: Handles `N % vec_width != 0` with scalar operations
6. **Correct return**: Exit block returns scalar from remainder loop, not vector from main loop

### Backend Implementation

Vector values are treated as first-class SSA values that:
- Get unique, never-reused stack slots (protected from slot recycler)
- Are excluded from GPR allocation (forced to stack)
- Use direct slot access in intrinsics (no pointer indirection)
- Support vector-to-vector Copy via ymm/xmm registers

### Generated Code (AVX2 Example)

```asm
vxorpd %ymm0, %ymm0, %ymm0          # Zero vector accumulator
.loop:
    vmovupd (%rax,%rcx), %ymm0      # Load 4 doubles
    vaddpd %ymm1, %ymm0, %ymm0      # Add 4 doubles
    ; loop back...

; Horizontal reduction
vextractf128 $1, %ymm0, %xmm1       # Extract high 128 bits
vaddpd %xmm1, %xmm0, %xmm0          # Add high + low (2 doubles each)
vunpckhpd %xmm0, %xmm0, %xmm1       # Unpack high double
vaddsd %xmm1, %xmm0, %xmm0          # Final scalar

; Remainder loop (scalar)
.remainder:
    movsd (%rbx,%r13,8), %xmm0      # Load single element
    addsd -24(%rbp), %xmm0          # Add to scalar accumulator
    ; loop back...

; Return scalar result
movsd -24(%rbp), %xmm0              # Return scalar (not vector!)
```

### Why It Beats GCC

GCC's auto-vectorizer is conservative on simple reductions:
- Worries about aliasing even with clear array indexing
- Considers the pattern "too simple" to benefit
- Falls back to 2× scalar loop unrolling

LCCC's pattern-based approach:
- Explicitly targets common reduction idioms
- Aggressively transforms when pattern matches
- Generates complete vectorization (4× for AVX2 vs GCC's 2× unroll)

**Result**: LCCC vectorizes patterns GCC -O3 leaves scalar, achieving ~2.7× speedup.

### Debug Flags

```bash
LCCC_DEBUG_VECTORIZE=1    # Show vectorization transformations
LCCC_DEBUG_PROTECT=1      # Show stack slot protection decisions
```

---
