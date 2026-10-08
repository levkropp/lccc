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
An 18-benchmark suite covering integer, FP, memory-bound, and call-heavy workloads. Best-of-9 wall-clock time; all 18 outputs byte-identical to GCC.

## Test Environment

| Item | Value |
|------|-------|
| **Host** | AArch64 Linux |
| **LCCC** | current `main` — linear scan + register steal + slot coalescing + NEON/F64 loop promotion + full SSA pipeline |
| **GCC** | system GCC, `-O2` |
| **Timing** | wall clock, 9 reps, best taken |
| **Correctness** | program output compared byte-for-byte against GCC |

## Results (LCCC / GCC -O2)

| Benchmark | Workload | LCCC | GCC | Ratio |
|-----------|----------|-----:|----:|:-----:|
| `arith_loop` | 32-var arithmetic loop | 0.034s | 0.030s | 1.12× |
| `fib` | fib(40) recursive | <0.001s | 0.090s | **~600× faster** |
| `matmul` | 256×256 FP matrix multiply | 0.0025s | 0.0029s | **1.15× faster** |
| `qsort` | quicksort 1M integers | 0.070s | 0.071s | **1.02× faster** |
| `sieve` | Eratosthenes 10M | 0.017s | 0.016s | 1.06× |
| `tce_sum` | tail-recursive sum(10M) | 0.0001s | 0.0001s | 1.0× |
| `nbody` | N-body FP simulation | 0.185s | 0.165s | 1.12× |
| `binary_trees` | malloc/free recursion | 0.848s | 0.751s | 1.13× |
| `spectral_norm` | FP dense loops | 0.133s | 0.133s | **1.0× parity** |
| `mandelbrot` | FP inner loop | 0.553s | 0.491s | 1.13× |
| `hash_table` | pointer chasing | 8.62s | 7.91s | 1.09× |
| `strlen_bench` | string processing | 0.145s | 0.140s | 1.04× |
| `switch_dispatch` | jump tables | 0.404s | 0.350s | 1.16× |
| `struct_copy` | struct copy/field access | 0.013s | 0.012s | 1.10× |
| `loop_patterns` | reduce/transform/prefix | 0.032s | 0.032s | **1.0× parity** |
| `fannkuch` | Fannkuch-Redux permutations | 2.010s | 1.973s | 1.02× |
| `ackermann` | ackermann(3,11) | <0.001s | 0.098s | **~600× faster** |
| `bitops` | popcount/clz/reverse | 0.089s | 0.169s | **1.9× faster** |

**Geometric mean: 0.49× of GCC -O2** across all 18 — skewed by the two ~600×
recursion-to-iteration wins; excluding `fib` and `ackermann`, **1.01× (statistical
parity)** across the remaining 16 (matmul, qsort, loop_patterns, bitops at or past
parity; spectral_norm, tce_sum, fannkuch, sieve, strlen_bench within 7%).

**Compile time:** LCCC compiles 2–5× faster than GCC across the suite.

## Where the Performance Comes From

### Recursion-to-iteration (`fib`, `ackermann` — ~600×)

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

fannkuch: 2.69× → 1.85×. spectral_norm: 1.63× → 1.19×.

### Loop-backedge slot coalescing (`arith_loop`)

A spilled loop-carried variable with a constant initializer used to keep a "double slot": the
phi dest and its backedge update each got a stack slot, with a `ldr`+`str` copy between them
on every iteration. The slot coalescer aliases the update into the phi dest's slot when the
phi-coalesce detector proves the old value is dead after the update is defined — the backedge
copy becomes a same-slot no-op. arith_loop: 2.09× → 1.15× (22 instructions/iteration gone).

### NEON vectorization and F64 loop promotion (`matmul`, `loop_patterns`, FP loops)

The inner loop of matmul is auto-vectorized with NEON (F64x2/I32x4 register-resident vector
ops, fused `fmadd`), F64 loop accumulators are promoted into dedicated FP registers
(d24–d31), and FP constants are hoisted out of loops. matmul runs 1.15× faster than GCC -O2.
Reduction vectorization covers plain and widening sums (`sadalp`), dot products
(`smlal`/`smlal2` with split accumulators), conditional sums (`smax` clamp), and max
reductions (`smax`/`smaxv`) — loop_patterns went from 1.83× to parity.

### FP recurrence scheduling and coalescing (`mandelbrot`, `nbody`)

Loop-phi anti-dependency splitting moves the old-value copy off the recurrence's serial
chain (the exact shape GCC -O2 emits), and reverse FP phi coalescing gives copy-web
accumulators the backedge source's register when conflict-free — the backedge `fmov`
dies. *Backedge PRE* then carries a loop-bottom expression (mandelbrot's `zr²`) into
the matching loop-top use through a new phi, with fusion-aware profitability so the
multiply absorbed into an `fmadd` stays fused — one `fmul` fewer per iteration, the
shape GCC reaches via loop rotation + CSE. mandelbrot 1.32× → 1.13×, nbody 1.31× → 1.12×.

### Conditional-increment fusion (`sieve`)

sieve's prime-count loop is dispatch/branch-throughput bound, not dependency bound —
and there, instruction count is everything. A `csinc` peephole folds the
increment-feeding-`csel` pair that if-conversion produces for `if (sieve[i]) count++`
into one instruction, cutting the loop from 12 instructions (~2 cycles/iter) toward
GCC's 5 (~0.65 cycles/iter). sieve 1.16× → 1.06×.

### Small-leaf caller-saved allocation and shrink-wrapping (`qsort`, `binary_trees`)

Call-free functions up to a few dozen IR instructions scan the caller-saved pool
(x4–x8, x13–x14) first: no prologue save storm for hot callbacks — qsort's `cmp`
callback dropped from 18 instructions plus five save/restore pairs to 7 instructions,
putting qsort at parity with GCC. Callee-saves are also shrink-wrapped past clean
early returns, so binary_trees' null-child path skips the restore storm.

### Bit manipulation (`bitops` — 1.8× faster than GCC)

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
