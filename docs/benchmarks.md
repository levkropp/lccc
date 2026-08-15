---
layout: doc
title: Benchmarks
description: LCCC vs GCC — methodology, results, and interpretation.
prev_page:
  title: Optimization Passes
  url: /docs/optimization-passes
next_page:
  title: Roadmap
  url: /docs/roadmap
---

# Benchmarks
{:.doc-subtitle}
An 18-benchmark suite covering integer, FP, memory-bound, and call-heavy workloads. Best-of-5 wall-clock time; all 18 outputs byte-identical to GCC.

## Test Environment

| Item | Value |
|------|-------|
| **Host** | AArch64 Linux |
| **LCCC** | current `main` — linear scan + register steal + slot coalescing + NEON/F64 loop promotion + full SSA pipeline |
| **GCC** | system GCC, `-O2` |
| **Timing** | wall clock, 5 reps, best taken |
| **Correctness** | program output compared byte-for-byte against GCC |

## Results (LCCC / GCC -O2)

| Benchmark | Workload | LCCC | GCC | Ratio |
|-----------|----------|-----:|----:|:-----:|
| `arith_loop` | 32-var arithmetic loop | 0.045s | 0.031s | 1.47× |
| `fib` | fib(40) recursive | <0.001s | 0.094s | **~500× faster** |
| `matmul` | 256×256 FP matrix multiply | 0.0029s | 0.0029s | **1.01× parity** |
| `qsort` | quicksort 1M integers | 0.081s | 0.072s | 1.13× |
| `sieve` | Eratosthenes 10M | 0.025s | 0.017s | 1.53× |
| `tce_sum` | tail-recursive sum(10M) | 0.0002s | 0.0002s | 1.0× |
| `nbody` | N-body FP simulation | 0.284s | 0.172s | 1.66× |
| `binary_trees` | malloc/free recursion | 0.934s | 0.772s | 1.21× |
| `spectral_norm` | FP dense loops | 0.175s | 0.137s | 1.28× |
| `mandelbrot` | FP inner loop | 0.820s | 0.498s | 1.65× |
| `hash_table` | pointer chasing | 12.01s | 9.18s | 1.31× |
| `strlen_bench` | string processing | 0.159s | 0.137s | 1.16× |
| `switch_dispatch` | jump tables | 0.444s | 0.346s | 1.28× |
| `struct_copy` | struct copy/field access | 0.027s | 0.012s | 2.31× |
| `loop_patterns` | reduce/transform/prefix | 0.059s | 0.032s | 1.83× |
| `fannkuch` | Fannkuch-Redux permutations | 3.82s | 1.97s | 1.94× |
| `ackermann` | ackermann(3,11) | <0.001s | 0.097s | **~500× faster** |
| `bitops` | popcount/clz/reverse | 0.103s | 0.172s | **1.7× faster** |

**Geometric mean: 0.64× of GCC -O2** across all 18 — skewed by the two ~500×
recursion-to-iteration wins; excluding `fib` and `ackermann`, **~1.34×** across the
remaining 16.

**Compile time:** LCCC compiles 2–5× faster than GCC across the suite.

## Where the Performance Comes From

### Recursion-to-iteration (`fib`, `ackermann` — ~500×)

Detects the `f(n) = f(n-1) + f(n-2)` binary-recursion pattern and converts the exponential
O(2ⁿ) call tree into an O(n) iterative sliding-window loop. GCC keeps the recursive calls.
This is a synthetic-benchmark win — no production code uses naive recursive Fibonacci — but it
demonstrates the pattern-matching infrastructure. Verified by a CI test computing fib(90),
impossible without the O(n) conversion.

### Register steal for hot loop values (`fannkuch`, `spectral_norm`)

The linear scan assigns registers in interval-start order and never evicts, so cold
function-spanning values (array bases, globals) used to win every callee-saved register
simply by starting first — leaving hot inner-loop-carried values (IVs, accumulators, carried
pointers) stack-homed. The post-scan steal picks, for each hot loop-carried phi value the
scan missed, the register whose *conflicting* holders have the coldest loop-weighted use
count, and fully deallocates those holders to the stack. Safe by construction (whole-interval
deallocation, no range splitting), and a no-op when the scan already housed the hot values.

fannkuch: 2.69× → 1.94×. spectral_norm: 1.63× → 1.28×.

### Loop-backedge slot coalescing (`arith_loop`)

A spilled loop-carried variable with a constant initializer used to keep a "double slot": the
phi dest and its backedge update each got a stack slot, with a `ldr`+`str` copy between them
on every iteration. The slot coalescer aliases the update into the phi dest's slot when the
phi-coalesce detector proves the old value is dead after the update is defined — the backedge
copy becomes a same-slot no-op. arith_loop: 2.09× → 1.47× (22 instructions/iteration gone).

### NEON vectorization and F64 loop promotion (`matmul`, FP loops)

The inner loop of matmul is auto-vectorized with NEON (F64x2/I32x4 register-resident vector
ops, fused `fmadd`), F64 loop accumulators are promoted into dedicated FP registers
(d24–d31), and FP constants are hoisted out of loops. matmul runs at parity with GCC -O2.

### Bit manipulation (`bitops` — 1.7× faster than GCC)

popcount/clz/ctz/bit-reverse lower directly to single AArch64 instructions (`cnt`+`uaddlv`,
`clz`, `rbit`), beating GCC's instruction selection on this workload.

### Tail-call elimination (`tce_sum`)

Self-recursive accumulator functions become counted loops — identical shape to GCC's output.
(GCC constant-folds `sum(10000000, 0)` entirely, so both are ~0s here.)

## Real-World: SQLite 3.45

LCCC compiles and fully runs the SQLite amalgamation (260K lines, single file): CREATE TABLE,
INSERT/UPDATE/DELETE, SELECT with WHERE/ORDER BY/LIMIT, JOINs, correlated and uncorrelated
subqueries, GROUP BY/HAVING, aggregates, UNION ALL, transactions, and prepared statements.

## Running the Benchmarks

```bash
# Full 18-benchmark suite (performance + correctness + size)
python3 tests/benchmark/run_benchmarks.py --reps 5

# A subset
python3 tests/benchmark/run_benchmarks.py --only fannkuch --only arith_loop --reps 8

# Compile a benchmark manually for disassembly comparison
GCC_INC="-I$(gcc -print-file-name=include)"
./target/release/lccc-arm $GCC_INC -O2 -o /tmp/arith_lccc tests/benchmark/programs/arith_loop.c
objdump -d /tmp/arith_lccc
```
