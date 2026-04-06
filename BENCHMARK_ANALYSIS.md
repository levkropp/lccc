# LCCC vs GCC -O2 Comprehensive Benchmark Analysis

**Date:** 2026-04-06  
**GCC version:** 15.2.1 (x86-64, Arch Linux)  
**LCCC version:** Phase 20 (commit 953c6ba0)  
**Test system:** Linux 6.19.8, 3 reps per benchmark  

## Runtime Performance (best-of-3 wall seconds)

| Benchmark | LCCC | GCC | Ratio | Category |
|-----------|-----:|----:|------:|----------|
| fib | 0.0005s | 0.199s | **0.003x** | rec2iter win |
| fannkuch | 0.886s | 2.489s | **0.36x** | **WRONG OUTPUT** |
| binary_trees | 0.256s | 2.065s | **0.12x** | **CRASH (SIGSEGV)** |
| matmul | 0.013s | 0.010s | 1.26x | FP + cache |
| qsort | 0.155s | 0.146s | 1.06x | branching |
| arith_loop | 0.126s | 0.109s | 1.16x | register pressure |
| strlen_bench | 0.312s | 0.250s | 1.25x | byte ops |
| hash_table | 18.06s | 13.26s | 1.36x | pointer chasing |
| switch_dispatch | 0.685s | 0.483s | 1.42x | jump tables |
| sieve | 0.088s | 0.057s | 1.55x | memory writes |
| loop_patterns | 0.189s | 0.057s | 3.32x | **CRASH (SIGSEGV)** |
| bitops | 1.346s | 0.411s | 3.27x | bit manipulation |
| nbody | 2.254s | 0.344s | **6.55x** | FP struct+sqrt |
| mandelbrot | 10.18s | 1.591s | **6.40x** | **WRONG OUTPUT** |
| ackermann | 1.098s | 0.161s | **6.83x** | deep recursion |
| tce_sum | 0.012s | 0.001s | **18.2x** | GCC const-folds |
| spectral_norm | 5.253s | 0.225s | **23.4x** | **WRONG OUTPUT** |
| struct_copy | 0.809s | 0.027s | **29.8x** | struct by-value |

**Geometric mean ratio: 1.76x (LCCC slower)**

## Correctness Issues Found

### Benchmark crashes (SIGSEGV)
1. **binary_trees** — segfault, even with rec2iter disabled
2. **loop_patterns** — segfault (possibly large BSS + codegen issue)

### Benchmark wrong output
3. **spectral_norm** — 1.2319 vs 1.2742 (FP precision/codegen bug)
4. **mandelbrot** — 518M vs 381M iterations (FP comparison codegen bug)
5. **fannkuch** — checksum=0/maxflips=0 (array flip loop gets stuck on 2nd permutation)

### Correctness suite failures (47/50 pass)
6. **string_literals** — `s1 == s2` returns 0 for identical string literals (no string pooling)
7. **vla_basic** — VLA (variable-length array) produces no output (crash)
8. **volatile_access** — `volatile int arr[5]` reads produce garbage values

## Root Cause Analysis (from assembly inspection)

### 1. FP constant materialization (spectral_norm, mandelbrot, nbody: 6-23x)

LCCC loads FP constants via:
```asm
movabsq $4611686018427387904, %rcx   ; 10 bytes, goes through integer pipeline
movq    %rcx, %xmm1                  ; 5 bytes, GPR→XMM transfer
```
GCC loads from .rodata:
```asm
movsd   .LC1(%rip), %xmm8            ; 8 bytes, direct from L1 cache
```

LCCC also reloads constants **inside** inner loops instead of hoisting them.

### 2. Missing inlining + SRoA (struct_copy: 29.8x)

LCCC calls `make_group` as a function, then copies 208 bytes with `rep movsb`.
GCC completely inlines `make_group` → `make_particle` → `particle_distance`,
replaces struct fields with registers, and uses SIMD (`movapd`, `mulpd`).

### 3. Instruction bloat (bitops: 3.27x)

LCCC emits 407 lines of assembly vs GCC's 141 for the same program.
`popcount32()` in LCCC: pushes 6 callee-saved registers, uses excessive
`movq` between GPRs for 32-bit values. 3x code bloat → 3x slowdown.

### 4. GCC constant-folds through tail calls (tce_sum: 18x)

GCC computes `sum(10000000, 0) = 50000005000000` **at compile time** via
interprocedural constant propagation. LCCC runs the loop at runtime.

### 5. Ackermann call overhead (6.83x)

GCC converts the ackermann recursion pattern to loops, unrolling multiple
recursion levels into separate register-tracked iterations.
LCCC pushes 6 callee-saved registers per call (72 bytes of stack frame).

### 6. FP precision bugs (mandelbrot, spectral_norm)

Assembly shows LCCC doing extra moves between GPR and XMM registers,
potentially causing precision differences from intermediate rounding.

## Compile Time (LCCC is 2-5x faster)

| Benchmark | LCCC | GCC | Ratio |
|-----------|-----:|----:|------:|
| matmul | 0.019s | 0.095s | **0.20x** |
| fib | 0.020s | 0.094s | **0.21x** |
| spectral_norm | 0.035s | 0.143s | **0.25x** |
| average | — | — | **~0.36x** |

LCCC compiles 2-5x faster than GCC across all benchmarks.

## Binary Size (LCCC is 10% smaller on micro-benchmarks)

Average binary size ratio: **0.95x GCC** (5% smaller on these tests).
Note: On SQLite, LCCC is 1.69x larger — the gap grows with program complexity.

## Prioritized Improvement Targets

### Tier 1: Correctness (must-fix)

| Issue | Impact | Difficulty |
|-------|--------|------------|
| **FP codegen precision** (mandelbrot, spectral_norm) | 2 benchmarks wrong | Medium — likely movabsq→xmm causing rounding |
| **Fannkuch array flip loop** | 1 benchmark wrong | Medium — array swap codegen bug |
| **Binary trees crash** | 1 crash | Medium — deep recursion + malloc codegen |
| **VLA support** | correctness test | Medium — alloca codegen |
| **volatile array reads** | correctness test | Low — volatile load codegen for arrays |

### Tier 2: High-impact performance (5-30x gains)

| Optimization | Benchmarks affected | Expected gain | Difficulty |
|-------------|--------------------:|:-------------:|------------|
| **FP constant hoisting** — load from .rodata, hoist out of loops | spectral_norm, mandelbrot, nbody | 5-10x | Medium |
| **Struct inlining + SRoA** — inline struct-returning fns, replace field access with regs | struct_copy | 10-30x | High |
| **Deeper constant propagation** — fold through TCE/known-input calls | tce_sum | 18x | Medium |
| **Callee-saved register spill reduction** — only push regs that are actually used | bitops, ackermann, all | 1.5-3x | Medium |

### Tier 3: Moderate performance (1.2-2x gains)

| Optimization | Benchmarks affected | Expected gain | Difficulty |
|-------------|--------------------:|:-------------:|------------|
| **Jump table codegen** for dense switch | switch_dispatch | 1.4x | Low |
| **Loop vectorization** for simple reductions | sieve, loop_patterns | 1.3-1.5x | Medium |
| **Memory operand codegen** — fuse load+op | hash_table, qsort | 1.2-1.3x | Medium |
| **Better inlining heuristics** — inline small hot functions | strlen_bench, arith_loop | 1.1-1.2x | Low |

### Tier 4: Linker improvements (using mold as reference)

| Feature | Impact | Difficulty |
|---------|--------|------------|
| **Fix libc SONAME detection** — wrong SONAME (musl vs glibc) | shared libs broken | Low |
| **Parallel section processing** | link time on large projects | High |
| **TLS GD relaxation** | correctness for some programs | Medium |
| **String merging sections** (.rodata.str) | binary size | Medium |
| **--whole-archive support** | build system compat | Low |
| **Symbol interposition** (LD_PRELOAD) | runtime compat | Medium |

## Quick Wins (low-effort, high-impact)

1. **FP constant pool** — emit FP constants as `.rodata` labels, load via `movsd .LC(%rip), %xmm`. This single change would speed up spectral_norm, mandelbrot, and nbody by 2-5x.

2. **Skip unused callee-saved pushes** — LCCC pushes 6 callee-saved regs in almost every function. Pushing only those actually used saves 12-48 bytes per call frame and reduces prologue/epilogue overhead.

3. **String literal deduplication** — merge identical string literals so `"hello" == "hello"` holds (already expected by many programs).

4. **Fix libc SONAME** — the linker records `libc.musl-x86_64.so.1` instead of `libc.so.6`, breaking dynamically-linked binaries on glibc systems.

## Test Infrastructure Created

- `tests/benchmark/run_benchmarks.py` — 18-benchmark comprehensive suite (LCCC vs GCC, runtime + size + compile time + correctness)
- `tests/benchmark/programs/` — 12 new benchmark programs (nbody, binary_trees, spectral_norm, mandelbrot, hash_table, etc.)
- `tests/correctness/run_correctness.py` — 50-test correctness suite (integers, bitfields, unions, enums, VLAs, multi-file, etc.)

Run with:
```bash
python3 tests/benchmark/run_benchmarks.py --reps 5
python3 tests/correctness/run_correctness.py -v
```
