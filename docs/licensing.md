---
layout: doc
title: Licensing
description: LCCC's dual-license model — CC0 for CCC-derived code, MIT/Apache/BSD for LCCC contributions.
prev_page:
  title: Roadmap
  url: /docs/roadmap
next_page:
---

# Licensing
{:.doc-subtitle}
LCCC uses a dual-license model to clearly separate original contributions from CCC-derived code.

## LCCC Contributions

All code authored as part of LCCC — new files, substantial rewrites, benchmark tools, documentation — is licensed under your choice of:

- **MIT** — [`LICENSE-MIT`](https://github.com/levkropp/lccc/blob/master/LICENSE-MIT)
- **Apache 2.0** — [`LICENSE-APACHE`](https://github.com/levkropp/lccc/blob/master/LICENSE-APACHE)
- **BSD 2-Clause** — [`LICENSE-BSD`](https://github.com/levkropp/lccc/blob/master/LICENSE-BSD)

This includes:
- `ccc/src/backend/live_range.rs` — the `LinearScanAllocator` and `LiveRange` data structures
- Changes to `ccc/src/backend/regalloc.rs` — the two-pass activation
- `lccc-improvements/benchmarks/bench.py` and benchmark C sources
- All documentation (`docs/`, `index.html`, `_layouts/`, etc.)

## CCC-Derived Code

All code derived from the original [CCC project](https://github.com/anthropics/claudes-c-compiler) is released under the **CC0 1.0 Universal** (public domain dedication):

- The CCC frontend (lexer, parser, semantic analysis)
- The SSA IR, mem2reg, and analysis infrastructure
- All optimization passes
- The x86-64, AArch64, RISC-V 64, and i686 backends
- The standalone assembler and linker

CCC was itself released as CC0 by Anthropic.

## Mixed Files

Some files in `ccc/src/backend/regalloc.rs` contain both CCC-original code and LCCC additions. In these files:
- Code present before LCCC's first commit is CC0
- Code added or substantially rewritten by LCCC is MIT/Apache/BSD

When in doubt, check `git log --follow -p <file>` to see which lines were introduced in which commit.

## Using LCCC in Your Project

If you want to use `live_range.rs` or the regalloc changes in your own project, the MIT license is the most permissive and easiest to comply with. Just keep the copyright notice.

If you need Apache 2.0 (e.g. for explicit patent grant), use that instead.

The CCC-derived portions have no restrictions — CC0 is effectively public domain.
