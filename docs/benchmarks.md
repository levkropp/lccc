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
Five micro-benchmarks targeting different bottlenecks. All measured with best-of-5 wall-clock time.

## Test Environment

| Item | Value |
|------|-------|
| **Host** | Linux x86-64 |
| **LCCC** | commit `ad9a287f` — linear scan active |
| **CCC** | commit `6f1b99ac` — upstream, three-phase allocator |
| **GCC** | 15.2.0 (Ubuntu 15.2.0-4ubuntu4) |
| **Flags** | `-O2` for all compilers |
| **Timing** | `time.perf_counter()` wall clock, 5 reps, best taken |
| **Date** | 2026-03-19 |

## Results

| Benchmark | LCCC | CCC | GCC -O2 | LCCC/CCC | LCCC/GCC |
|-----------|-----:|----:|--------:|:--------:|:--------:|
| `arith_loop` | **0.124s** | 0.149s | 0.068s | **+20% faster** | 1.83× slower |
| `sieve` | **0.036s** | 0.045s | 0.025s | **+25% faster** | 1.46× slower |
| `qsort` | 0.096s | 0.095s | 0.087s | ≈ equal | 1.10× slower |
| `fib(40)` | 0.352s | 0.354s | 0.095s | ≈ equal | 3.70× slower |
| `matmul` | 0.028s | 0.029s | 0.003s | ≈ equal | 7.91× slower |

All outputs are byte-identical to GCC's.

## Benchmark Descriptions

### `01_arith_loop` — Register Pressure

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

**LCCC vs CCC:** 0.124s vs 0.149s (**+20% faster**). Stack frame shrinks from 0x1f0 (496B) to slightly larger (more prologue saves) but with fewer memory ops in the hot loop body.

**LCCC vs GCC:** 1.83× slower. GCC allocates all 32 values across caller- and callee-saved registers simultaneously (it has a graph-coloring allocator), while LCCC's two-pass approach has more interference between the passes.

### `02_fib` — Recursive Calls

```c
long fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}
// fib(40) = 102,334,155  (~330M recursive calls)
```

**Why it matters:** Call-dominated workload. Every call site clobbers caller-saved registers, making callee-saved registers the only useful ones. Both LCCC and CCC allocate identically here — the function body is too small to benefit from better allocation.

**LCCC vs CCC:** ≈ equal (0.352s vs 0.354s). Both compile `fib` to the same two-recursive-call structure.

**LCCC vs GCC:** 3.70× slower. GCC eliminates the second recursive call via a loop transformation (tail-call optimization for the accumulated variant), dramatically reducing call overhead.

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

**LCCC vs CCC:** ≈ equal (0.028s vs 0.029s). Both emit scalar `mulsd`/`addsd`. The register allocator's integer improvements don't apply to XMM registers.

**LCCC vs GCC:** 7.91× slower. GCC auto-vectorizes the inner loop with AVX2 `vfmadd231pd`, processing 4 doubles per instruction. This is a Phase 3+ improvement target.

### `04_qsort` — Library Calls

```c
qsort(arr, N, sizeof(int), cmp);
```

**Why it matters:** Minimal compiler involvement — `qsort` is a libc function. The only LCCC code is the comparison function and the setup.

**LCCC vs CCC:** ≈ equal (0.096s vs 0.095s). Expected — `cmp` is trivial.

**LCCC vs GCC:** 1.10× slower. Close to parity because libc does the work.

### `05_sieve` — Memory Writes + Inner Loop

```c
for (int i = 2; i*i <= N; i++)
    if (sieve[i])
        for (int j = i*i; j <= N; j += i)
            sieve[j] = 0;
```

**Why it matters:** The inner loop has two variables (`j`, `i`) that should stay in registers. Both are integer loop variables — exactly what the allocator targets.

**LCCC vs CCC:** 0.036s vs 0.045s (**+25% faster**). LCCC keeps `j` and `i` in registers across the inner loop; CCC reloads them from the stack each iteration.

**LCCC vs GCC:** 1.46× slower. GCC unrolls and uses `rep stosd`-style patterns; the remaining gap is loop unrolling and branch prediction.

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
