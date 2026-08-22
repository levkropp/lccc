---
layout: doc
title: Roadmap
description: Where LCCC stands vs GCC and what's next.
prev_page:
  title: Benchmarks
  url: /docs/benchmarks
next_page:
  title: Licensing
  url: /docs/licensing
---

# Roadmap
{:.doc-subtitle}
LCCC's optimization work is organized in phases. Phases 1–20 are complete; current work is driven directly by the benchmark suite.

## Where We Stand

18-benchmark suite vs GCC -O2 (AArch64, best-of-5, all outputs byte-identical):

- **1.06× of GCC -O2** across the 16 conventional benchmarks (0.50× geomean including the
  two ~600× recursion-to-iteration wins)
- **matmul 1.15× faster**, **bitops 1.8× faster**, **qsort and spectral_norm faster**,
  **loop_patterns at parity**, **fannkuch 1.03×**
- SQLite 3.45 amalgamation compiles and runs correctly (260K lines)

## What's Next (driven by the suite)

| Benchmark | Gap | Plan |
|-----------|-----|------|
| `hash_table` | 1.22× | Pointer-chasing bound; load/store tightening plateau reached |
| `sieve` / `switch_dispatch` / `arith_loop` | ~1.15× | Residual slot traffic; needs the slot-home rewrite |
| `mandelbrot` / `nbody` | ~1.12× | Copy-web accumulators keep slot homes across loop nests |
| `binary_trees` | 1.14× | Recursive inlining (GCC unrolls self-calls ~5 levels) |
| Correctness | 1 + 2 suite failures | designated_init (correctness), L6 matrix_multiply (progressive) |

Longer-term candidates: slot-home elimination (on-demand spilling — the documented
structural rewrite), better function inlining (~1.5× on call-heavy code), instruction
scheduling (~1.1× on latency-bound code), profile-guided optimization (~1.2–1.5× general).

The goal is not to beat GCC — it's to make CCC-compiled programs fast enough for real
systems software. Typical workloads now run within ~1.06× of GCC -O2.

## What's Been Done

A compressed history of the landed phases (git log has the full record):

| Phases | Work | Headline result |
|--------|------|-----------------|
| 1–2 | Register allocation analysis + linear-scan allocator | +20–25% on register-pressure code |
| 3 | Tail-call elimination + phi-copy slot coalescing | 139× on accumulator recursion |
| 4–5 | Loop unrolling, FP intrinsic lowering, FP peephole | Matmul −41% time |
| 6–8 | SSE2 → AVX2 vectorization, remainders, reductions | ~2.7× faster than GCC -O3 on reductions |
| 9–11 | Indexed addressing (SIB), const stores, accumulator folding | Sieve 1.78× → 1.55× |
| 12–13 | Regalloc loop-depth fix, sign-ext fusion, loop rotation | Sieve 6-instruction inner loop |
| 14–15 | Correctness hardening (36+ bugs), peephole re-enablement | Full SQLite works |
| 16–18 | Register-direct codegen, vectorizer fixes, MachInst ISel | −78KB then −41.7KB .text on SQLite |
| 19–20 | Live range splitting, ISel expansion + encoding fixes | 1.69× GCC binary size on SQLite |
| Recent | AArch64: NEON vector ops, F64 loop promotion, register steal, loop-backedge slot coalescing, full reduction vectorization (sums, dots, conditional sums, max), int-widen register casts, FP anti-dependency splitting + reverse phi coalescing, small-leaf caller-saved allocation, shrink-wrap, several latent miscompile fixes | Geomean 0.86× → 0.50× of GCC |

Notable dead-ends (documented so they don't get retried): in-scan register eviction
(miscompiles), loop-transparent live-range splitting (behind `CCC_LOOP_SPLIT`; in-loop
values can't be transparently split), unconditional register pre-pinning (pool shrinkage
evicts warm values — superseded by the post-scan steal).

## Historical Write-ups

Dated phase write-ups and posts are kept for the record:

- [Phase 3: TCE + phi coalescing](/lccc/updates/phase3-tce-and-phi-coalescing)
- [Phase 4: loop unrolling](/lccc/updates/phase4-loop-unrolling)
- [Phase 5: FP peephole](/lccc/updates/phase5-fp-peephole)
- Posts: [SSE2 vectorization](/lccc/docs/_posts/2026-03-20-phase-6-sse2-vectorization),
  [remainder loops](/lccc/docs/_posts/2026-03-20-phase-7b-remainder-loops),
  [SQLite works](/lccc/docs/_posts/2026-03-27-phase-14-sqlite-works)
