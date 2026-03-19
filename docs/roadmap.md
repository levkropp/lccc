---
layout: doc
title: Roadmap
description: LCCC optimization phases — what's done, what's next.
prev_page:
  title: Benchmarks
  url: /docs/benchmarks
next_page:
  title: Licensing
  url: /docs/licensing
---

# Roadmap
{:.doc-subtitle}
LCCC improves CCC in five phases. Phase 2 is complete.

## Status Overview

| Phase | Name | Status | Expected gain |
|-------|------|--------|---------------|
| 1 | Register allocator analysis & design | ✅ Complete | (prerequisite) |
| 2 | Linear scan register allocator | ✅ Complete | **+20–25% on register-pressure code** |
| 3 | Loop unrolling & branch optimization | 🔲 Planned | ~1.5–2× on loop-heavy code |
| 4 | SIMD / auto-vectorization | 🔲 Planned | ~4–8× on FP-heavy code |
| 5 | Profile-guided optimization (PGO) | 🔲 Planned | ~1.2–1.5× general |

---

## Phase 1 — Register Allocator Analysis (Complete)

**Goal:** Understand the current allocator, measure its limitations, and design a replacement.

**Deliverables:**
- `CURRENT_ALLOCATOR_ANALYSIS.md` — deep analysis of the three-phase greedy allocator (350 lines)
- `LINEAR_SCAN_DESIGN.md` — complete algorithm specification (500 lines)
- `INTEGRATION_POINTS.md` — exact call sites, data flow, interface requirements (350 lines)

**Key findings:**
- The old allocator only considers ~5% of IR values eligible (very conservative whitelist)
- No interval splitting or spill-cost-based eviction — all-or-nothing allocation
- Phase 1 excluded non-call-spanning values from callee-saved registers entirely
- 32-variable function: 11KB stack frame, 0 registers allocated (all spilled)

---

## Phase 2 — Linear Scan Register Allocator (Complete)

**Goal:** Replace the three-phase greedy allocator with a proper linear scan.

**What changed** (`ccc/src/backend/`):
- `live_range.rs` — new: `LiveRange`, `ActiveInterval`, `LinearScanAllocator` (796 lines)
- `regalloc.rs` — `allocate_registers()` now runs two-pass linear scan

**Algorithm:** Poletto & Sarkar (1999) linear scan with priority-based eviction:
1. Build enhanced live ranges with loop-depth-weighted priorities
2. Process intervals in order of start point
3. Expire ended intervals, freeing their registers
4. Assign a free register or evict the lowest-weight active interval
5. Second pass: assign caller-saved registers to unallocated non-call-spanning values

**Results:**
- `arith_loop` (32 vars): 0.124s vs 0.149s — **+20% faster** than CCC
- `sieve`: 0.036s vs 0.045s — **+25% faster** than CCC
- 500 unit tests pass; all benchmark outputs identical to GCC

---

## Phase 3 — Loop Optimizations (Planned)

**Goal:** Close the gap on loop-heavy integer code via unrolling and branch-prediction-aware transforms.

**Techniques:**
- **Loop unrolling** — replicate loop body 2–4× to reduce branch overhead and expose ILP
- **Loop interchange** — improve cache locality on nested loops (important for matmul)
- **Tail duplication** — duplicate small blocks to eliminate backward branches
- **Branch-to-cmov** — extend if-conversion to more patterns

**Expected impact:**
- `fib(40)`: currently 3.70× slower than GCC; loop-based fib variant could reach ~1.5×
- `arith_loop`: could reach ~1.2× with 4× unroll reducing loop overhead
- `sieve` inner loop: unrolling eliminates ~25% of branch mispredictions

**Implementation target:** `passes/loop_unroll.rs` (new), extend `passes/if_convert.rs`

---

## Phase 4 — SIMD / Auto-Vectorization (Planned)

**Goal:** Emit AVX2/SSE4 instructions for vectorizable inner loops.

**Current gap:** `matmul` is 7.91× slower than GCC because GCC emits `vfmadd231pd` (AVX2 FMA, 4 doubles/cycle) while LCCC emits scalar `mulsd`/`addsd` (1 double/cycle).

**Techniques:**
- **SLP vectorization** — pack scalar ops in a basic block into SIMD ops
- **Loop vectorization** — vectorize simple counted loops with compatible access patterns
- **NEON/RVV/SSE backends** — architecture-specific vector lowering

**Implementation target:** new `passes/vectorize.rs`, extended x86/ARM/RISC-V backends

**Estimated gain:** 4–8× on FP-heavy code; closes the matmul gap from 7.91× to ~1.5–2×

---

## Phase 5 — Profile-Guided Optimization (Planned)

**Goal:** Use runtime profiles to guide inlining, block layout, and register allocation priorities.

**Techniques:**
- **Instrumented build** → collect branch frequencies, call counts, loop trip counts
- **Cold code separation** — move unlikely blocks out of hot paths
- **PGO-weighted inlining** — inline hot callees aggressively, skip cold ones
- **PGO register hints** — weight allocator priorities by actual execution frequency

**Expected gain:** ~1.2–1.5× general improvement across diverse workloads

---

## Remaining Gap to GCC

Even after all five phases, some gaps will remain:

| Source | Gap | Addressable? |
|--------|-----|-------------|
| SIMD vectorization | 4–8× on FP loops | Phase 4 |
| Loop unrolling | 1.3–2× on tight loops | Phase 3 |
| Graph-coloring register allocation | ~1.1× on register-pressure code | Future Phase 6 |
| Link-time optimization (LTO) | ~1.1× general | Future |
| Whole-program devirtualization | negligible for C | N/A |

The goal is not to beat GCC — it's to make CCC-compiled programs fast enough for real systems software, which means within ~1.5× of GCC on typical workloads.
