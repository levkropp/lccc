---
layout: doc
title: Architecture
description: How LCCC relates to CCC, the compilation pipeline, and what LCCC changes.
prev_page:
  title: Getting Started
  url: /docs/getting-started
next_page:
  title: Register Allocator
  url: /docs/register-allocator
---

# Architecture
{:.doc-subtitle}
LCCC is a fork of CCC. The core compilation pipeline is unchanged; LCCC replaces and improves specific components.

## Relationship to CCC

[CCC (Claude's C Compiler)](https://github.com/anthropics/claudes-c-compiler) is a C compiler written from scratch in Rust. It implements the full toolchain — frontend, SSA IR, optimizer, code generators for four architectures, assembler, and linker — with zero external dependencies.

LCCC started as a fork tracked as a git submodule; the sources are now developed in-tree. The compiler lives in `src/`, the benchmark/test suites in `tests/`, and this site in `docs/` + `index.html`.

```
lccc/
├── src/
│   ├── frontend/       ← lexer, preprocessor, parser, sema
│   ├── ir/             ← SSA IR, mem2reg, analysis
│   ├── passes/         ← 27 optimization passes + loop analysis
│   ├── backend/        ← code generation, regalloc, assembler, linker (4 targets)
│   ├── common/         ← shared types, diagnostics
│   └── driver/         ← CLI, pipeline orchestration
├── include/            ← bundled C headers (SSE–AVX-512, NEON, …)
├── tests/              ← benchmark / correctness / integration / regression suites
├── index.html          ← this site (landing page)
└── docs/               ← this site (documentation)
```

## Compilation Pipeline

```
C source
   │
   ▼  frontend/
   │  lexer → preprocessor → parser → sema → IR lowering
   │
   ▼  ir/
   │  mem2reg (alloca → SSA phi nodes), dominators, loop analysis
   │
   ▼  passes/  (27 passes, up to three full iterations)
   │  tce · rec2iter · inline · GVN · LICM · IVSR · unroll · vectorize
   │  if-conv · DCE · const-fold · copy-prop · narrow · divconst · bit-idioms …
   │
   ▼  backend/  (per-architecture)
   │  ├── liveness.rs      — backward-dataflow live intervals + loop depths
   │  ├── live_range.rs    — LinearScanAllocator
   │  ├── regalloc.rs      — pools, phi register coalescing, loop-value steal
   │  ├── stack_layout/    — tiered slot allocation, copy/slot coalescing
   │  ├── <target>/        — instruction selection + emission + peephole
   │  ├── assembler        — standalone (no external toolchain)
   │  └── linker           — ELF executable writer
   │
   ▼
ELF executable
```

## What LCCC Changes

### Register allocation

CCC's allocator uses three greedy phases with a conservative eligibility whitelist (~5% of IR values). LCCC replaces the allocation core with a linear scan over live intervals:

| | Old (CCC) | New (LCCC) |
|---|---|---|
| **Algorithm** | Greedy priority sort, no eviction | Linear scan + post-scan steal for hot loop values |
| **Phase 1** | Callee-saved for call-spanning values only | Callee-saved for all eligible values |
| **Phase 2** | Caller-saved for non-call-spanning values | Caller-saved for unallocated non-call-spanning values |
| **FP values** | Always stack-homed | FP/SIMD pool (AArch64 d16–d31) for F64 accumulators and vectors |
| **Loop values** | Lose to early-starting cold values | Steal registers from provably colder holders |
| **Spill decision** | Skip the value | Skip, then rebalance hot loop-carried phi values |

The eligibility filter — which excludes floats (GPR pool), i128, atomic pointers, memcpy pointers, and VA arg pointers — is unchanged. It is the correctness boundary between safe and unsafe register allocation.

### Stack layout

A three-tier slot allocator: permanent slots for addressable allocas, liveness-packed slots for multi-block values, and block-local greedy reuse for short-lived values. Copy coalescing (including loop-backedge phi slot coalescing) makes phi-elimination copies same-slot no-ops.

### Peephole optimizers

Each backend runs a text-level peephole pipeline over emitted assembly — store/load forwarding, dead-move elimination, sign-extension fusion, memory-operand folding, indexed addressing. The optimizers are deliberately conservative; several latent miscompiles were found and fixed by auditing them.

### Licensing Model

LCCC uses a dual-license approach:

- **LCCC contributions** (new code, analysis, benchmarks): MIT OR Apache-2.0 OR BSD-2-Clause
- **CCC-derived code** (frontend, SSA IR, optimizer, backends, assembler, linker): CC0 1.0

When a file contains both, both licenses apply to their respective portions.

## Architecture-Agnostic Register Allocation

The allocator works through a small, stable interface:

```rust
pub struct RegAllocConfig {
    pub available_regs:    Vec<PhysReg>,  // callee-saved
    pub caller_saved_regs: Vec<PhysReg>,  // caller-saved
    pub xmm_regs:          Vec<PhysReg>,  // FP/SIMD pool
    pub allow_inline_asm_regalloc: bool,
}

pub fn allocate_registers(func: &IrFunction, config: &RegAllocConfig) -> RegAllocResult;
```

Each architecture backend calls `allocate_registers` with its own register lists. `PhysReg(n)` is just a numeric index — the allocator never knows which architecture it is running on.

| Architecture | Callee-saved pool | Caller-saved pool | FP/SIMD pool |
|---|---|---|---|
| x86-64 | rbx, r12–r15 | r10, r11, r8, r9 | xmm2–xmm7 (F64) |
| AArch64 | x19–x28 (10 regs) | x4–x8, x13, x14 (7 regs) | d16–d31 (F64/vector, loop promotion) |
| RISC-V 64 | s1, s7–s11 | (varies) | — |
| i686 | ebx, esi, edi | — | — |
