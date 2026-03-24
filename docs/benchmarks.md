---
layout: doc
title: Benchmarks
description: LCCC vs CCC vs GCC — methodology, results, and interpretation.
prev_page:
  title: Optimization Passes
  url: /docs/optimization-passes
next_page:
  title: Roadmap
  url: /docs/roadmap
---

# Benchmarks
{:.doc-subtitle}
Six micro-benchmarks targeting different bottlenecks. All measured with best-of-5 wall-clock time.

## Test Environment

| Item | Value |
|------|-------|
| **Host** | Linux x86-64 |
| **LCCC** | Phase 13 complete — linear scan + TCE + phi-copy coalescing + FP opts + AVX2 vectorization + reduction vectorization + rec-to-iter + SIB fold + accumulator fold + regalloc loop-depth fix + sign-ext fusion + phi-copy chain coalescing + loop rotation |
| **CCC** | upstream, three-phase greedy allocator |
| **GCC** | 15.2.1 (Arch Linux) |
| **Flags** | `-O2` for all compilers (GCC `-O3 -march=native` for reduction comparison) |
| **Timing** | `time.perf_counter()` wall clock, 7 reps, best taken |
| **Date** | 2026-03-24 |

## Results

| Benchmark | LCCC | GCC -O2 | LCCC/GCC |
|-----------|-----:|--------:|:--------:|
| `arith_loop` | 0.131s | 0.082s | 1.60× slower |
| `sieve` | 0.048s | 0.044s | **1.09× slower** |
| `qsort` | 0.122s | 0.101s | 1.20× slower |
| `fib(40)` | **0.001s** | 0.136s | **478× faster** |
| `matmul` | 0.008s | 0.005s | 1.60× slower |
| `reduction` | **AVX2** | scalar (GCC -O3) | **~2.7× faster** |
| `tce_sum` | 0.007s | 0.001s | 7× slower (GCC const-folds) |

All outputs are byte-identical to GCC's.

## Benchmark Descriptions

### `01_arith_loop` — Register Pressure + Phi-Copy Coalescing

```c
// 32 local int variables, all updated every loop iteration, 10M iterations
int arith_loop(int n) {
    int a=1, b=2, c=3, ..., z7=32;
    for (int i = 0; i < n; i++) {
        a += b*c; b += c*d; ...  // 32 cross-dependent updates
    }
    return a ^ b ^ c ^ ... ^ z7;
}
```

**Why it matters:** This is the canonical register allocation stress test. With 32 live integer variables and 10M iterations, every extra stack spill costs ~3ns. LCCC's linear scan assigns more callee-saved registers to the hottest values.

**LCCC vs CCC:** 0.104s vs 0.147s (**+41% faster**). Two improvements combine here:
1. **Linear-scan register allocation** (Phase 2): keeps more variables in callee-saved registers across the loop
2. **Phi-copy stack coalescing** (Phase 3b): phi elimination creates one Copy instruction per loop variable per backedge, generating ~20 redundant stack-to-stack `movq` pairs per iteration. Reversing the alias direction (src borrows dest's wider-live slot) makes these copies same-slot no-ops, dropped by the code generator

**LCCC vs GCC:** 1.53× slower. GCC allocates all 32 values across caller- and callee-saved registers simultaneously (graph-coloring allocator), while LCCC's two-pass approach has more interference between the passes.

### `02_fib` — Recursive Calls

```c
long fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}
// fib(40) = 102,334,155  (~330M recursive calls)
```

**Why it matters:** Call-dominated workload — tests whether the compiler can recognize and optimize the exponential recursion pattern.

**LCCC vs GCC: 478× faster.** LCCC's binary recursion-to-iteration pass (Phase 10) detects the `f(n) = f(n-1) + f(n-2)` pattern and converts it to an O(n) iterative sliding-window loop. GCC -O2 keeps the exponential recursion (with partial loop transformation of one call). The transformation is verified by a CI test that computes fib(90) — impossible without the O(n) conversion.

**Note:** This is a synthetic benchmark. No production code uses naive recursive Fibonacci. The optimization demonstrates LCCC's pattern-matching capabilities but should not be interpreted as "LCCC is faster than GCC" in general — GCC wins on all other benchmarks.

### `03_matmul` — Floating Point + Cache

```c
void matmul(void) {
    for (int i = 0; i < 256; i++)
      for (int k = 0; k < 256; k++)
        for (int j = 0; j < 256; j++)
          C[i][j] += A[i][k] * B[k][j];
}
```

**Why it matters:** FP throughput and cache behavior. The inner loop is a scalar FP multiply-add.

**LCCC vs CCC:** ≈ equal (0.027s vs 0.029s). Both emit scalar `mulsd`/`addsd`. The register allocator's integer improvements don't apply to XMM registers.

**LCCC vs GCC:** 7.84× slower. GCC auto-vectorizes the inner loop with AVX2 `vfmadd231pd`, processing 4 doubles per instruction. This is a Phase 5 improvement target.

### `04_qsort` — Library Calls

```c
qsort(arr, N, sizeof(int), cmp);
```

**Why it matters:** Minimal compiler involvement — `qsort` is a libc function. The only LCCC code is the comparison function and the setup.

**LCCC vs CCC:** ≈ equal (0.098s vs 0.096s). Expected — `cmp` is trivial.

**LCCC vs GCC:** 1.13× slower. Close to parity because libc does the work.

### `05_sieve` — Memory Writes + Inner Loop

```c
for (int i = 2; i*i <= N; i++)
    if (sieve[i])
        for (int j = i*i; j <= N; j += i)
            sieve[j] = 0;
```

**Why it matters:** The inner loop has two variables (`j`, `i`) that should stay in registers. Both are integer loop variables — exactly what the allocator targets.

**LCCC vs CCC:** 0.037s vs 0.044s (**+19% faster**). LCCC keeps `j` and `i` in registers across the inner loop; CCC reloads them from the stack each iteration.

**LCCC vs GCC:** 1.54× slower. GCC uses branchless counting (`sbb` trick) and loop unrolling; the remaining gap is branch prediction and loop overhead.

### `06_reduction` — Reduction Vectorization (NEW)

```c
double sum_array(double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
// sum_array(array, 10000000)
```

**Why it matters:** Simple reduction patterns are common in scientific computing but surprisingly hard to auto-vectorize. GCC's vectorizer is conservative and often leaves them scalar.

**LCCC vs CCC:** **4× speedup**. LCCC detects the reduction pattern and transforms to AVX2 SIMD (4 doubles per iteration), with proper horizontal reduction (`vextractf128` + `vunpckhpd` + `vaddsd`) and a remainder loop for N % 4 != 0. CCC emits scalar `addsd`.

**LCCC vs GCC -O3:** **~2.7× faster**. This is LCCC's signature win: GCC -O3 with `-march=native` does NOT vectorize this pattern—it only does 2× scalar loop unrolling. LCCC generates 12 ymm instructions; GCC generates 0. Assembly comparison:
```asm
; LCCC -O2
vxorpd %ymm0, %ymm0, %ymm0          # Zero vector
vmovupd (%rax,%rcx), %ymm0          # Load 4 doubles
vaddpd %ymm1, %ymm0, %ymm0          # Add 4 doubles
vextractf128 $1, %ymm0, %xmm1       # Horizontal reduction
vunpckhpd %xmm0, %xmm0, %xmm1
vaddsd %xmm1, %xmm0, %xmm0          # Final scalar

; GCC -O3 -march=native
vxorpd %xmm0, %xmm0, %xmm0          # Scalar zero
vaddsd (%rdi), %xmm0, %xmm0         # Scalar add
vaddsd -8(%rdi), %xmm0, %xmm0       # Scalar add (unrolled)
```

### `07_tce_sum` — Tail-Call Elimination

```c
static long sum(int n, long acc) {
    if (n <= 0) return acc;
    return sum(n - 1, acc + n);  // tail call
}
// sum(10000000, 0) = 50000005000000
```

**Why it matters:** A pure accumulator-style recursion with 10M stack frames in CCC. LCCC's tail-call elimination (TCE) converts the self-recursive tail call to a back-edge branch before the main optimization loop, turning 10M stack frames into a tight loop.

**LCCC vs CCC:** 0.008s vs 1.09s (**139× faster**). CCC executes 10M actual `call`/`ret` pairs; LCCC emits a counted loop identical to what GCC produces.

**LCCC vs GCC:** ≈ equal. Both emit a 3-instruction counted loop — LCCC's TCE matches GCC's output here.

## Running the Benchmarks

```bash
# Build LCCC
cargo build --release

# Run full suite (requires upstream CCC at ../ccc-upstream or adjust paths)
python3 lccc-improvements/benchmarks/bench.py --reps 5 --md results.md

# Run a single benchmark, more reps
python3 lccc-improvements/benchmarks/bench.py --bench 01_arith_loop --reps 20

# Compile a benchmark manually for disassembly comparison
GCC_INC="-I/usr/lib/gcc/x86_64-linux-gnu/$(gcc -dumpversion)/include"
./target/release/ccc $GCC_INC -O2 -o /tmp/arith_lccc lccc-improvements/benchmarks/bench/01_arith_loop.c
objdump -d /tmp/arith_lccc | grep -A 100 "arith_loop"
```
