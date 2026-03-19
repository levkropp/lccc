# LCCC Benchmark Report

Generated: 2026-03-19 18:44:54

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
| **Arith Loop** | 32-var arithmetic, 10M iters  (register pressure) | 0.124s | 0.149s | 0.068s | 1.83× | 2.20× | 14 KB | 15 KB |
| **Fibonacci** | fib(40) recursive              (call overhead) | 0.352s | 0.354s | 0.095s | 3.70× | 3.73× | 14 KB | 15 KB |
| **MatMul 256²** | 256×256 double matrix multiply (FP + cache) | 0.028s | 0.029s | 0.003s | 7.91× | 8.23× | 14 KB | 15 KB |
| **Quicksort 1M** | sort 1M integers               (branching) | 0.096s | 0.095s | 0.087s | 1.10× | 1.09× | 14 KB | 15 KB |
| **Sieve 10M** | primes to 10M                  (memory writes) | 0.036s | 0.045s | 0.025s | 1.46× | 1.82× | 14 KB | 15 KB |

## Correctness

| Benchmark | LCCC == GCC | CCC == GCC |
|-----------|:-----------:|:----------:|
| Arith Loop | ✅ | ✅ |
| Fibonacci | ✅ | ✅ |
| MatMul 256² | ✅ | ✅ |
| Quicksort 1M | ✅ | ✅ |
| Sieve 10M | ✅ | ✅ |

## Notes

- LCCC uses the new linear-scan register allocator (Phase 2 of the LCCC
  optimization roadmap). The allocator runs two passes:
  1. **Callee-saved pass**: linear scan over all eligible IR values,
     assigning callee-saved registers (safe across calls).
  2. **Caller-saved pass**: linear scan over unallocated, non-call-spanning
     values, assigning caller-saved registers.
- CCC uses the original three-phase greedy allocator.
- GCC `-O2` is the performance baseline.

