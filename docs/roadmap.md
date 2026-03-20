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
LCCC improves CCC in six phases. Phases 1–3 are complete.

## Status Overview

| Phase | Name | Status | Result |
|-------|------|--------|--------|
| 1 | Register allocator analysis & design | ✅ Complete | (prerequisite) |
| 2 | Linear scan register allocator | ✅ Complete | **+20–25% on register-pressure code** |
| 3a | Tail-call-to-loop elimination (TCE) | ✅ Complete | **139× on accumulator recursion** |
| 3b | Phi-copy stack slot coalescing | ✅ Complete | **+19% additional on loop-heavy code** |
| 4 | Loop unrolling & branch optimization | 🔲 Planned | ~1.5–2× on loop-heavy code |
| 5 | SIMD / auto-vectorization | 🔲 Planned | ~4–8× on FP-heavy code |
| 6 | Profile-guided optimization (PGO) | 🔲 Planned | ~1.2–1.5× general |

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

**What changed** (`src/backend/`):
- `live_range.rs` — new: `LiveRange`, `ActiveInterval`, `LinearScanAllocator` (796 lines)
- `regalloc.rs` — `allocate_registers()` now runs two-pass linear scan

**Algorithm:** Poletto & Sarkar (1999) linear scan with priority-based eviction:
1. Build enhanced live ranges with loop-depth-weighted priorities
2. Process intervals in order of start point
3. Expire ended intervals, freeing their registers
4. Assign a free register or evict the lowest-weight active interval
5. Second pass: assign caller-saved registers to unallocated non-call-spanning values

**Results:**
- `arith_loop` (32 vars): 0.124s vs 0.149s — **+20% faster** than CCC (before Phase 3b)
- `sieve`: 0.037s vs 0.044s — **+19% faster** than CCC
- 500 upstream tests pass; all benchmark outputs identical to GCC

---

## Phase 3a — Tail-Call Elimination (Complete)

**Goal:** Convert self-recursive tail calls to back-edge branches, eliminating stack frames
for accumulator-style recursive functions.

**What changed** (`src/passes/`):
- `tail_call_elim.rs` — new: `tail_calls_to_loops()` pass (747 lines)
- `mod.rs` — TCE runs once after inlining, before the main optimization loop

**Algorithm:** A tail call is a recursive call whose result is returned immediately with no
further computation. TCE converts it to a loop by:
1. Inserting a loop header block with one Phi per parameter
2. Renaming parameter references in the body to use the Phi outputs
3. Replacing the tail call + return with assignments to the Phi inputs + unconditional branch

The resulting loop is then optimized by LICM, IVSR, and GVN in the normal pipeline.

**Results:**
- `tce_sum`: 0.008s vs 1.09s — **139× faster** than CCC (matches GCC exactly)
- Zero regressions on 508 unit tests

**Pass name:** `tce` (disable with `CCC_DISABLE_PASSES=tce`)

---

## Phase 3b — Phi-Copy Stack Slot Coalescing (Complete)

**Goal:** Eliminate the redundant stack-to-stack copies that phi elimination creates for
spilled loop variables.

**Root cause:** When CCC's phi elimination lowers SSA phi nodes to Copy instructions, each
loop variable `%i` gets a separate copy per predecessor:
- Entry: `%i = Copy 0`
- Backedge: `%i = Copy %i_next`

Because `%i` is now "multi-defined", the existing copy coalescing refused to share slots
between `%i` and `%i_next`. This produced ~20 redundant `movq mem, %rax; movq %rax, mem`
pairs per iteration in a 32-variable loop.

**Fix:** For the phi-copy pattern — where `%i_next` is defined and killed in the backedge
block (sole use = the Copy), and `%i` is used in other blocks (the loop header) — alias in
the *reversed* direction: `%i_next` borrows `%i`'s wider-live slot. The backedge copy
becomes a same-slot no-op and is dropped by `generate_copy`.

The key insight: `%i` being multi-defined only means it's the slot *owner* (written in
multiple predecessors). The *aliased* value `%i_next` is always single-defined, making the
alias safe.

**What changed** (`src/backend/stack_layout/copy_coalescing.rs`):
- Moved `multi_def_values.contains(&d)` check from the early combined guard to after the
  phi-copy pattern branch — multi-def is fine for the slot owner, never for the aliased value

**Results:**
- `arith_loop`: 0.124s → **0.104s** (+19% additional speedup on top of Phase 2)
- Combined Phase 2+3b vs CCC: **+41% faster** on the 32-variable loop benchmark

---

## Phase 4 — Loop Optimizations (Planned)

**Goal:** Close the gap on loop-heavy integer code via unrolling and branch-prediction-aware transforms.

**Techniques:**
- **Loop unrolling** — replicate loop body 2–4× to reduce branch overhead and expose ILP
- **Loop interchange** — improve cache locality on nested loops (important for matmul)
- **Tail duplication** — duplicate small blocks to eliminate backward branches
- **Branch-to-cmov** — extend if-conversion to more patterns

**Expected impact:**
- `arith_loop`: could reach ~1.2× with 4× unroll reducing loop overhead
- `sieve` inner loop: unrolling eliminates ~25% of branch mispredictions

**Implementation target:** `passes/loop_unroll.rs` (new), extend `passes/if_convert.rs`

---

## Phase 5 — SIMD / Auto-Vectorization (Planned)

**Goal:** Emit AVX2/SSE4 instructions for vectorizable inner loops.

**Current gap:** `matmul` is 7.84× slower than GCC because GCC emits `vfmadd231pd` (AVX2 FMA, 4 doubles/cycle) while LCCC emits scalar `mulsd`/`addsd` (1 double/cycle).

**Techniques:**
- **SLP vectorization** — pack scalar ops in a basic block into SIMD ops
- **Loop vectorization** — vectorize simple counted loops with compatible access patterns
- **NEON/RVV/SSE backends** — architecture-specific vector lowering

**Implementation target:** new `passes/vectorize.rs`, extended x86/ARM/RISC-V backends

**Estimated gain:** 4–8× on FP-heavy code; closes the matmul gap from 7.84× to ~1.5–2×

---

## Phase 6 — Profile-Guided Optimization (Planned)

**Goal:** Use runtime profiles to guide inlining, block layout, and register allocation priorities.

**Techniques:**
- **Instrumented build** → collect branch frequencies, call counts, loop trip counts
- **Cold code separation** — move unlikely blocks out of hot paths
- **PGO-weighted inlining** — inline hot callees aggressively, skip cold ones
- **PGO register hints** — weight allocator priorities by actual execution frequency

**Expected gain:** ~1.2–1.5× general improvement across diverse workloads

---

## Remaining Gap to GCC

Even after all six phases, some gaps will remain:

| Source | Gap | Addressable? |
|--------|-----|-------------|
| SIMD vectorization | 4–8× on FP loops | Phase 5 |
| Loop unrolling | 1.3–2× on tight loops | Phase 4 |
| Graph-coloring register allocation | ~1.1× on register-pressure code | Future Phase 7 |
| Link-time optimization (LTO) | ~1.1× general | Future |
| Whole-program devirtualization | negligible for C | N/A |

The goal is not to beat GCC — it's to make CCC-compiled programs fast enough for real systems software, which means within ~1.5× of GCC on typical workloads.
