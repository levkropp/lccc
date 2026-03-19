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

LCCC is a fork tracked as a git submodule. The `ccc/` directory contains the compiler source; `lccc-improvements/` contains analysis, benchmarks, and documentation for the improvements. Changes are made in the submodule and tested against the full upstream test suite before landing.

```
lccc/
├── ccc/                    ← git submodule (compiler source, CC0 licensed)
│   ├── src/
│   │   ├── frontend/       ← lexer, parser, type-checker
│   │   ├── ir/             ← SSA IR, mem2reg, analysis
│   │   ├── passes/         ← optimizer: GVN, LICM, IPCP, DCE, inliner…
│   │   └── backend/        ← code generation, regalloc, assembler, linker
│   └── Cargo.toml
├── lccc-improvements/
│   ├── register-allocation/   ← Phase 1 design docs
│   └── benchmarks/            ← bench.py + C benchmark sources
├── index.html              ← this site (landing page)
├── docs/                   ← this site (documentation)
└── Cargo.toml              ← workspace root
```

## Compilation Pipeline

```
C source
   │
   ▼  frontend/
   │  ├── lexer.rs       — tokenize
   │  ├── parser.rs      — build AST (recursive descent)
   │  └── codegen_ir.rs  — lower AST → SSA IR
   │
   ▼  ir/
   │  ├── mem2reg.rs     — promote alloca → SSA phi nodes
   │  └── analysis/      — dominator tree, loop analysis, liveness
   │
   ▼  passes/  (optimizer — all levels run the same pipeline)
   │  ├── inline.rs      — function inlining
   │  ├── gvn.rs         — global value numbering / CSE
   │  ├── licm.rs        — loop-invariant code motion
   │  ├── ipcp.rs        — interprocedural constant propagation
   │  ├── dce.rs         — dead code elimination
   │  ├── constant_fold  — constant folding
   │  ├── copy_prop      — copy propagation
   │  └── cfg_simplify   — branch threading, dead block removal
   │
   ▼  backend/  (per-architecture)
   │  ├── regalloc.rs    ← LCCC: two-pass linear scan (replaces greedy)
   │  ├── live_range.rs  ← LCCC: LiveRange, LinearScanAllocator
   │  ├── liveness.rs    — backward-dataflow live interval computation
   │  ├── generation.rs  — instruction selection + emission
   │  ├── peephole       — architecture-specific strength reduction
   │  ├── stack_layout/  — stack frame layout after regalloc
   │  ├── elf/           — ELF object file writer
   │  └── linker_common/ — standalone linker
   │
   ▼
ELF executable
```

## What LCCC Changes

### Phase 2: Register Allocator (complete)

**File:** `ccc/src/backend/regalloc.rs`, `ccc/src/backend/live_range.rs`

The old allocator uses three greedy phases with a conservative eligibility whitelist (~5% of IR values). LCCC replaces the allocation core with a two-pass linear scan:

| | Old (CCC) | New (LCCC) |
|---|---|---|
| **Algorithm** | Greedy priority sort | Linear scan with eviction |
| **Phase 1** | Callee-saved for call-spanning values only | Callee-saved for all eligible values |
| **Phase 2** | Caller-saved for non-call-spanning values | Caller-saved for unallocated non-call-spanning values |
| **Phase 3** | Callee-saved spillover | — (folded into Phase 1) |
| **Spill decision** | Just skip the value | Evict lowest-weight active interval |
| **Eligibility filter** | Kept intact (correctness boundary) | Kept intact (same rules) |

The eligibility filter — which excludes floats, i128, atomic pointers, memcpy pointers, and VA arg pointers — is unchanged. It is the correctness boundary between safe and unsafe register allocation.

### Licensing Model

LCCC uses a dual-license approach:

- **LCCC contributions** (new code, analysis, benchmarks): MIT OR Apache-2.0 OR BSD-2-Clause
- **CCC-derived code** (the `ccc/` submodule): CC0 1.0 (public domain dedication)

When a file contains both, both licenses apply to their respective portions.

## Architecture-Agnostic Register Allocation

The allocator works through a small, stable interface:

```rust
pub struct RegAllocConfig {
    pub available_regs:        Vec<PhysReg>,  // callee-saved
    pub caller_saved_regs:     Vec<PhysReg>,  // caller-saved
    pub allow_inline_asm_regalloc: bool,
}

pub fn allocate_registers(func: &IrFunction, config: &RegAllocConfig) -> RegAllocResult;
```

Each architecture backend (x86, ARM, RISC-V, i686) calls `allocate_registers` with its own register list. `PhysReg(n)` is just a numeric index — the allocator never knows which architecture it is running on.

| Architecture | Callee-saved available | Caller-saved available |
|---|---|---|
| x86-64 | rbx, r12–r15 (4–5 regs) | r10, r11, r8, r9 (4 regs) |
| AArch64 | x20–x28 (up to 9 regs) | x13, x14 (2 regs) |
| RISC-V 64 | s1, s7–s11 (6 regs) | (varies) |
| i686 | ebx, esi, edi (3 regs) | — |
