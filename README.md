# LCCC — Lev's Claude's C Compiler

> An optimized fork of [CCC](https://github.com/anthropics/claudes-c-compiler): a zero-dependency
> C compiler in Rust with its own assembler and linker, targeting x86-64, AArch64, RISC-V 64,
> and i686. LCCC adds a linear-scan register allocator with hot-loop register steal, phi
> coalescing for both registers and stack slots, SIMD auto-vectorization (AVX2/SSE2 on x86-64,
> NEON on AArch64), tail-call and recursion-to-iteration transforms, and per-target peephole
> optimizers. **~1.1× of GCC -O2** on the 18-benchmark suite (AArch64), **600× faster** on
> naive recursive benchmarks, and it compiles real software like SQLite.

**[Documentation](https://levkropp.github.io/lccc/)** ·
**[Benchmarks](#benchmarks)** ·
**[Getting Started](#getting-started)** ·
**[Roadmap](#roadmap)**

---

## What is LCCC?

CCC (Claude's C Compiler) is a zero-dependency C compiler written entirely in Rust by Claude Opus 4.6,
capable of compiling real projects — PostgreSQL, SQLite, Redis, the Linux kernel — with its own
assembler and linker (no external toolchain).

LCCC is a performance fork. The original CCC shuttles every value through the accumulator with no
register allocation; LCCC replaces that with a linear-scan register allocator and adds the
optimization machinery needed to approach GCC -O2: SSA optimization pipeline, SIMD
vectorization, loop transforms, and target-specific peephole optimizers.

```
C source
  │  frontend: lex → parse → sema → IR lowering
  ▼
SSA IR
  │  optimizer: TCE · rec2iter · inline · GVN · LICM · IVSR · unroll · vectorize
  │             · if-conv · DCE · const-fold · narrow · divconst · bit-idioms (27 passes)
  ▼
Optimized IR
  │  regalloc: linear scan over live intervals
  │    · callee-saved pool: all eligible values, start-order
  │    · caller-saved pool: non-call-spanning values
  │    · FP/SIMD pool: F64 accumulators, NEON/SSE vector values
  │    · post-scan steal: hot loop-carried phi values evict cold holders
  ▼
Machine code  (x86-64 · AArch64 · RISC-V 64 · i686)
  │  stack layout: liveness-packed slots · loop-backedge phi slot coalescing
  │  peephole: target-specific text passes (FP round-trips, mem folds, copy prop, …)
  │  standalone assembler + linker (no external toolchain)
  ▼
ELF executable
```

---

## Benchmarks

18-benchmark suite, LCCC vs GCC -O2, best-of-5 wall-clock, AArch64 Linux (same-run ratios).
All 18 outputs are byte-identical to GCC.

| Benchmark | LCCC | GCC -O2 | LCCC / GCC |
|-----------|-----:|--------:|:----------:|
| `arith_loop` — 32-var arithmetic loop | 0.035 s | 0.030 s | 1.15× |
| `fib` — fib(40) recursive | <0.001 s | 0.091 s | **~600× faster** |
| `matmul` — 256×256 matrix multiply | 0.0024 s | 0.0029 s | **1.17× faster** |
| `qsort` — quicksort 1M integers | 0.080 s | 0.071 s | 1.12× |
| `sieve` — Eratosthenes 10M | 0.019 s | 0.016 s | 1.15× |
| `tce_sum` — tail-recursive sum(10M) | 0.0001 s | 0.0001 s | 1.0× (GCC const-folds it) |
| `nbody` — N-body simulation | 0.218 s | 0.166 s | 1.31× |
| `binary_trees` — malloc/free recursion | 0.908 s | 0.755 s | 1.20× |
| `spectral_norm` — FP dense loops | 0.165 s | 0.139 s | 1.19× |
| `mandelbrot` — FP inner loop | 0.663 s | 0.502 s | 1.32× |
| `hash_table` — pointer chasing | 11.47 s | 9.16 s | 1.25× |
| `strlen_bench` — string processing | 0.160 s | 0.139 s | 1.16× |
| `switch_dispatch` — jump tables | 0.418 s | 0.348 s | 1.20× |
| `struct_copy` — struct copy/field access | 0.014 s | 0.011 s | 1.20× |
| `loop_patterns` — reduce/transform/prefix | 0.030 s | 0.031 s | **1.0× (parity)** |
| `fannkuch` — Fannkuch-Redux | 3.34 s | 1.81 s | 1.85× |
| `ackermann` — ackermann(3,11) | <0.001 s | 0.098 s | **~600× faster** |
| `bitops` — popcount/clz/reverse | 0.092 s | 0.169 s | **1.8× faster** |

**Geometric mean: 0.55× of GCC -O2** overall — skewed by the two ~600×
recursion-to-iteration wins; excluding those, **1.12×** across the remaining 16.

What drives the wins:

- **fib / ackermann ~600×**: binary recursion-to-iteration converts exponential O(2^n)
  recursion into an O(n) iterative sliding-window loop. GCC keeps the recursive calls.
- **matmul 1.17× faster**: NEON FMA vectorization on AArch64 (AVX2 FMA3 on x86-64), with
  loop-aware inlining and F64 accumulators promoted into FP registers across the inner loop.
- **loop_patterns 1.83× → parity** (recent work): NEON reduction vectorization now covers
  plain sums, i32→i64 widening sums (sadalp), dot products (smlal/smlal2 with split
  accumulators), conditional sums (`smax` clamp), and max reductions (`smax`/`smaxv`).
- **fannkuch 2.7× → 1.85×, arith_loop 2.1× → 1.15×**: a post-scan *register
  steal* gives hot inner-loop-carried values (IVs, accumulators, carried pointers) callee-saved
  registers that cold function-spanning values would otherwise win purely by starting first —
  done as a conflict-safe full deallocation of the evicted holder, never range splitting.
  *Loop-backedge slot coalescing* gives a spilled loop variable's update the variable's own
  stack slot (proven safe by the phi-coalesce detector), deleting the per-iteration
  `ldr`+`str` shuffle.
- **struct_copy 2.31× → 1.20×**: aggregate memcpy-temp forwarding, full unrolling of small
  constant-trip loops, and ldp/stp adjacent-field pair fusion.
- **bitops 1.8× faster than GCC**: popcount/clz/reverse lower to single AArch64 instructions.

**Compile time:** LCCC compiles **2–5× faster** than GCC across all benchmarks.

### Running the suites

```bash
# Full 18-benchmark suite (performance + correctness + size)
python3 tests/benchmark/run_benchmarks.py --reps 5

# 50-test correctness suite
python3 tests/correctness/run_correctness.py

# Progressive feature tests (levels 1–8)
python3 tests/integration/test_progressive.py
```

---

## Real-World Compatibility

LCCC compiles and **fully runs** the SQLite amalgamation (260K lines, single file):
CREATE TABLE, INSERT/UPDATE/DELETE, SELECT with WHERE/ORDER BY/LIMIT, JOINs, correlated and
uncorrelated subqueries, GROUP BY/HAVING, aggregates, UNION ALL, transactions, and prepared
statements all work.

An [independent benchmark](https://github.com/harshavmb/compare-claude-compiler) found the
original CCC was 737× slower than GCC on SQLite. LCCC addresses every major issue identified:

| CCC issue | LCCC status |
|-----------|-------------|
| No register allocation (shuttle %rax) | Linear scan, three register pools + loop-value steal |
| No optimization tiers (-O2 = -O0) | Full SSA optimization pipeline at -O2 |
| No function inlining | Loop-aware inlining pass with heuristics |
| No vectorization | AVX2/SSE2 on x86-64, NEON on AArch64 |
| 2.78× code bloat | Down to ~1.7× GCC .text on SQLite |
| 737× slower runtime | ~1.1× of GCC -O2 on the benchmark suite |
| Corrupted frame pointers | CFI directives, frame pointer omission |

Known correctness gaps are tracked by the test suites (5 failures in the 50-test correctness
suite, 3 in the progressive suite — struct passing, designated initializers, multi-dim arrays,
string arrays, typedef complexity, void-pointer arithmetic).

---

## Inside LCCC

### Linear-scan register allocation

CCC's three-phase greedy allocator considered only ~5% of IR values eligible; a 32-variable
function got **zero** registers. LCCC runs a Poletto & Sarkar linear scan
([`src/backend/live_range.rs`](src/backend/live_range.rs),
[`src/backend/regalloc.rs`](src/backend/regalloc.rs)) over live intervals:

- **Callee-saved pool** for all eligible values; **caller-saved pool** for values that don't
  span a call; **FP/SIMD pool** for F64 accumulators and vector values (AArch64: d16–d31).
- Uses are weighted by loop depth (10^depth), so inner-loop temporaries outrank straight-line
  values.
- **Post-scan register steal** (AArch64): hot loop-carried phi values the scan missed take a
  register from the coldest conflicting holder, which is fully deallocated to the stack —
  safe where in-scan eviction was not.
- **Phi register coalescing**: a loop-carried phi and its backedge source share a register,
  making the backedge copy a no-op.

### Phi-copy stack-slot coalescing

Spilled loop variables suffered a "double slot" problem: the phi dest and its backedge update
each got a stack slot, with a load+store shuffle between them every iteration. The slot
coalescer ([`src/backend/stack_layout/copy_coalescing.rs`](src/backend/stack_layout/copy_coalescing.rs))
aliases the update into the phi dest's slot when the phi-coalesce detector proves the old value
is dead — the backedge copy becomes a same-slot no-op. Multi-source phi webs (switch joins) are
handled by a separate interference-checked phase.

### SIMD vectorization

- **x86-64**: SSE2 2-wide and AVX2 4-wide (FMA3) auto-vectorization for matmul-style loops and
  reductions, with remainder loops for any trip count.
- **AArch64**: NEON intrinsics and register-based F64x2/I32x4 vector ops, FP constant hoisting,
  and F64 loop accumulators promoted into dedicated FP registers (d24–d31).

### Loop transforms

Tail-call elimination turns self-recursive tail calls into loops; recursion-to-iteration
handles the binary-recursive cases (fib, ackermann). Loop unrolling, rotation, strength
reduction (IVSR), LICM, and loop memory promotion run in the main SSA pipeline.

### Peephole optimizers

Each backend runs a text-level peephole pipeline over the emitted assembly (store/load
forwarding, dead-move elimination, sign-extension fusion, memory-operand folding, indexed
addressing). The optimizers are deliberately conservative — several latent miscompiles were
found and fixed by auditing these passes.

---

## Getting started

**Prerequisites:** Rust stable (2021 edition), a POSIX host.

```bash
git clone https://github.com/levkropp/lccc.git
cd lccc
cargo build --release
# → target/release/lccc        (native target picked by host/argv[0])
# → target/release/lccc-x86    (x86-64)
# → target/release/lccc-arm    (AArch64)
# → target/release/lccc-riscv  (RISC-V 64)
# → target/release/lccc-i686   (i686)

# Compile a C file (GCC's built-in headers are needed)
GCC_INC="-I/usr/lib/gcc/x86_64-linux-gnu/$(gcc -dumpversion)/include"
./target/release/lccc $GCC_INC -O2 -o hello hello.c
./hello

# Use as a drop-in GCC replacement
make CC=/path/to/target/release/lccc
```

For cross-compilation targets install the matching sysroot
(`aarch64-linux-gnu-gcc`, `riscv64-linux-gnu-gcc`). The target is selected by argv[0]:
symlink or rename the binary (e.g. `aarch64-linux-gnu-cc`) to select that target.

### GCC-compatible flags

```bash
lccc -S input.c                    # emit assembly
lccc -c input.c                    # compile to object file
lccc -O2 -o output input.c         # optimize (-O0 through -O3, -Os, -Oz)
lccc -g -o output input.c          # DWARF debug info
lccc -DFOO=1 -Iinclude/ input.c    # macros + include paths
lccc -fPIC -shared -o lib.so lib.c # position-independent code
```

Unrecognized flags are silently ignored so `lccc` works as a drop-in in build systems.

### Environment variables

| Variable | Effect |
|----------|--------|
| `CCC_TIME_PASSES` | Print per-pass timing and change counts to stderr |
| `CCC_DISABLE_PASSES` | Disable passes by name (comma-separated, or `all`) |
| `CCC_KEEP_ASM` | Keep intermediate `.s` files next to output |
| `LCCC_DEBUG_VECTORIZE` | Print vectorization pattern matching details |

Individual backend features have `CCC_NO_*` kill switches (e.g. `CCC_NO_LOOP_PIN`,
`CCC_NO_LOOP_PHI_SLOT`, `CCC_NO_PEEPHOLE`, `CCC_NO_SLOT_COALESCE`) used for A/B benchmarking
and miscompile bisection; grep for `CCC_` in `src/backend` for the full list.

Pass names: `cfg`, `copyprop`, `narrow`, `simplify`, `constfold`, `gvn`, `licm`,
`ifconv`, `dce`, `ipcp`, `inline`, `ivsr`, `divconst`, `tce`, `unroll`, `vectorize`.

---

## Project layout

```
src/
  frontend/     C source → typed AST (preprocessor, lexer, parser, sema)
  ir/           Target-independent SSA IR (lowering, mem2reg, analysis)
  passes/       SSA optimization passes (27 passes + shared loop analysis)
  backend/      IR → assembly → ELF (4 architectures)
    live_range.rs     LinearScanAllocator
    regalloc.rs       pools, phi register coalescing, loop-value steal
    stack_layout/     tiered slot allocation, copy/slot coalescing
    arm/ x86/ riscv/ i686/   per-target codegen + peephole
  common/       Shared types, symbol table, diagnostics
  driver/       CLI parsing, pipeline orchestration

include/        Bundled C headers (SSE–AVX-512, AES-NI, FMA, SHA, BMI2; NEON)
tests/          benchmark/ correctness/ integration/ regression/ suites

docs/           Jekyll documentation site source
updates/        Historical optimization write-ups (kept for the record)
```

---

## Roadmap

Current priorities, driven by the benchmark suite:

- **fannkuch (1.85×)** — loop-nested live-range splitting: inner-loop scalars spill because
  outer-loop values pin the GPR pool across the whole body
- **mandelbrot / nbody (~1.3×)** — FP codegen: fewer GPR↔FP round-trips, better scheduling
- **hash_table (1.25×)** — pointer-chasing load/store tightening
- Correctness: close the 5 remaining correctness-suite failures (struct passing, designated
  initializers, multi-dim arrays)

The goal is not to beat GCC — it's to make CCC-compiled programs fast enough for real systems
software. Typical workloads now run within **~1.12× of GCC -O2**.

---

## Testing

```bash
# Unit tests (563 pass, 6 ignored)
cargo test --release

# Correctness suite (50 programs, vs GCC output)
python3 tests/correctness/run_correctness.py

# Progressive integration tests (levels 1–8)
python3 tests/integration/test_progressive.py

# Benchmark suite (18 programs, correctness + performance + size)
python3 tests/benchmark/run_benchmarks.py --reps 5
```

---

## Licensing

LCCC uses a dual-license model to separate original contributions from CCC-derived code.

**LCCC contributions** (new files, regalloc changes, benchmarks, docs) —
MIT OR Apache-2.0 OR BSD-2-Clause (your choice). See `LICENSE-MIT`, `LICENSE-APACHE`, `LICENSE-BSD`.

**CCC-derived code** (frontend, SSA IR, optimizer, backends, assembler, linker) —
CC0 1.0 Universal (public domain). CCC was released as CC0 by Anthropic.

See [`LICENSING.md`](LICENSING.md) for the full breakdown and per-file guidance.
