# LCCC Benchmark Report

Generated: 2026-03-19 20:47:02

## Compilers

| Name | Binary |
|------|--------|
| LCCC | `/home/min/lccc/target/release/ccc` (linear-scan allocator active) |
| CCC  | `/home/min/ccc-upstream/target/release/ccc` (upstream, three-phase allocator) |
| GCC  | `/usr/bin/gcc` (`-O2`) |

## Results

All times are **best-of-N wall-clock seconds** (lower is better).  
Ratios are relative to GCC -O2.

| Benchmark | Description | LCCC | CCC | GCC | LCCC/GCC | CCC/GCC | Size LCCC | Size GCC |
|-----------|-------------|-----:|----:|----:|---------:|--------:|----------:|---------:|
| **Arith Loop** | 32-var arithmetic, 10M iters  (register pressure) | 0.103s | 0.150s | 0.068s | 1.52× | 2.21× | 14 KB | 15 KB |
| **Fibonacci** | fib(40) recursive              (call overhead) | 0.353s | 0.353s | 0.095s | 3.72× | 3.72× | 14 KB | 15 KB |
| **MatMul 256²** | 256×256 double matrix multiply (FP + cache) | 0.027s | 0.028s | 0.004s | 6.00× | 6.36× | 14 KB | 15 KB |
| **Quicksort 1M** | sort 1M integers               (branching) | 0.096s | 0.094s | 0.087s | 1.10× | 1.08× | 14 KB | 15 KB |
| **Sieve 10M** | primes to 10M                  (memory writes) | 0.036s | 0.044s | 0.025s | 1.45× | 1.80× | 14 KB | 15 KB |

## Correctness

| Benchmark | LCCC == GCC | CCC == GCC |
|-----------|:-----------:|:----------:|
| Arith Loop | ✅ | ✅ |
| Fibonacci | ✅ | ✅ |
| MatMul 256² | ✅ | ✅ |
| Quicksort 1M | ✅ | ✅ |
| Sieve 10M | ✅ | ✅ |

## Notes

- LCCC uses the new linear-scan register allocator (Phase 2) + phi-copy coalescing
  (Phase 3b) + tail-call elimination (Phase 3a) + loop unrolling (Phase 4).
  - **Loop unrolling** (`src/passes/loop_unroll.rs`): unrolls small inner loops
    (≤8 body-work blocks, ≤60 instructions) with an "early-exit" strategy that
    handles non-multiple trip counts without a cleanup loop. Factors: 8×/4×/2×.
    Only applies to loops with a single latch, constant-step IV, and detectable
    exit condition. The sieve inner loop (`j += i`, variable step) is NOT unrolled;
    the sieve counting loop is unrolled 8×.
  - **Register allocator**: two-pass linear scan (callee-saved + caller-saved).
  - **Phi-copy coalescing** (`src/passes/phi_copy_coalesce.rs`): eliminates
    redundant copies at phi join points.
  - **Tail-call elimination** (Phase 3a): converts tail-recursive calls to loops.
- CCC uses the original three-phase greedy allocator (no loop opts).
- GCC `-O2` is the performance baseline.

