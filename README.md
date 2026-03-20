# LCCC — Lev's Claude's C Compiler

> An optimized fork of [CCC](https://github.com/anthropics/claudes-c-compiler) with a two-pass
> linear-scan register allocator. **+20–25% faster** on register-pressure code vs upstream.

**[Documentation](https://levkropp.github.io/lccc/)** ·
**[Benchmarks](#benchmarks)** ·
**[Getting Started](#getting-started)** ·
**[Roadmap](#roadmap)**

---

## What is LCCC?

CCC (Claude's C Compiler) is a zero-dependency C compiler written entirely in Rust by Claude Opus 4.6,
capable of compiling real projects — PostgreSQL, SQLite, Redis, the Linux kernel — for x86-64, AArch64,
RISC-V 64, and i686, with its own assembler and linker.

LCCC is a performance fork. Phase 2 replaces CCC's three-phase greedy register allocator with a proper
two-pass linear-scan allocator (Poletto & Sarkar 1999), yielding +20–25% speedups on
register-pressure code while keeping all 500 upstream tests green.

```
C source
  │  frontend: lex → parse → sema → IR lowering
  ▼
SSA IR
  │  optimizer: GVN · LICM · IPCP · DCE · const-fold · inline
  ▼
Optimized IR
  │  regalloc (LCCC): two-pass linear scan over live intervals
  │    pass 1: callee-saved  ↔ all eligible values
  │    pass 2: caller-saved  ↔ non-call-spanning unallocated values
  ▼
Machine code  (x86-64 · AArch64 · RISC-V 64 · i686)
  │  standalone assembler + linker (no external toolchain)
  ▼
ELF executable
```

---

## Benchmarks

Best-of-5 wall-clock, Linux x86-64, GCC 15.2.0, `-O2` for all compilers.
All outputs are byte-identical to GCC.

| Benchmark | LCCC | CCC | GCC -O2 | vs CCC | vs GCC |
|-----------|-----:|----:|--------:|:------:|:------:|
| `arith_loop` — 32-var arithmetic, 10 M iters | **0.124 s** | 0.149 s | 0.068 s | **+20% faster** | 1.83× slower |
| `sieve` — primes to 10 M | **0.036 s** | 0.045 s | 0.025 s | **+25% faster** | 1.46× slower |
| `qsort` — sort 1 M integers | 0.096 s | 0.095 s | 0.087 s | ≈ equal | 1.10× slower |
| `fib(40)` — recursive Fibonacci | 0.352 s | 0.354 s | 0.095 s | ≈ equal | 3.70× slower |
| `matmul` — 256×256 double | 0.028 s | 0.029 s | 0.003 s | ≈ equal | 7.91× slower |
| `tce_sum` — tail-recursive sum(10M) | **0.008 s** | 1.09 s | 0.008 s | **139× faster** | ≈ equal |

The gains on `arith_loop` and `sieve` come directly from keeping more loop variables in registers.
The `tce_sum` gain comes from tail-call elimination: LCCC converts the 10M recursive calls into a
loop, matching GCC. CCC executes 10M actual stack frames.
The `matmul` gap is GCC's AVX2 auto-vectorization — a Phase 4 target.

Run the suite yourself:

```bash
python3 lccc-improvements/benchmarks/bench.py --reps 5 --md results.md
```

---

## Linear-scan register allocator

CCC's original allocator uses three greedy phases and only considers ~5% of IR values eligible
(a conservative whitelist). A 32-variable function allocates **0 registers** — every value spills
to an 11 KB stack frame.

LCCC replaces the allocation core with Poletto & Sarkar linear scan:

**Priority:** `use_count × 10^loop_depth` — values used inside deep loops get priority.

**Spill weight:** `priority / interval_length` — shorter, hotter intervals beat longer, cooler ones.

**Pass 1 — callee-saved registers (all eligible values):**

```rust
let ranges    = build_live_ranges(&intervals, &loop_depth, func);
let mut alloc = LinearScanAllocator::new(ranges, config.available_regs.clone());
alloc.run();  // expire → find_free → evict-lowest-weight-or-spill
```

**Pass 2 — caller-saved registers (unallocated, non-call-spanning values):**

```rust
let phase2 = intervals.iter()
    .filter(|iv| !assignments.contains_key(&iv.value_id))
    .filter(|iv| !spans_any_call(iv, call_points))
    .collect();
let mut caller_alloc = LinearScanAllocator::new(phase2, config.caller_saved_regs.clone());
caller_alloc.run();
```

Caller-saved registers are destroyed by calls, so only values whose live range doesn't cross any
call site are eligible for Pass 2. This is both correct and sufficient — call-heavy code like
recursive `fib` sees no change, which is expected.

The allocator lives in [`src/backend/live_range.rs`](src/backend/live_range.rs) (796 lines).
See the [register allocator docs](https://levkropp.github.io/lccc/docs/register-allocator) for
the full algorithm walk-through.

---

## Tail-call elimination

Self-recursive tail calls are converted to back-edge branches (loops), eliminating stack frame
overhead for accumulator-style functions. A tail call is a recursive call whose result is returned
immediately with no further computation.

```c
// Before TCE: 10M stack frames
long sum(int n, long acc) {
    if (n <= 0) return acc;
    return sum(n - 1, acc + n);  // tail call → loop back-edge
}
```

After TCE, the IR grows a loop header with one Phi node per parameter, and the call+return becomes
an unconditional branch back to the header. Subsequent passes (LICM, IVSR, GVN) then optimize the
resulting loop. On `sum(10000000, 0)`: LCCC finishes in 8 ms; CCC takes 1.09 s.

The pass lives in [`src/passes/tail_call_elim.rs`](src/passes/tail_call_elim.rs) and runs once
after inlining, before the main optimization loop.

---

## Getting started

**Prerequisites:** Rust stable (2021 edition), Linux x86-64 host.

```bash
# Clone
git clone https://github.com/levkropp/lccc.git
cd lccc

# Build
cargo build --release
# → target/release/ccc       (x86-64)
# → target/release/ccc-arm   (AArch64)
# → target/release/ccc-riscv (RISC-V 64)
# → target/release/ccc-i686  (i686)

# Compile a C file (GCC built-in headers are needed)
GCC_INC="-I/usr/lib/gcc/x86_64-linux-gnu/$(gcc -dumpversion)/include"
./target/release/ccc $GCC_INC -O2 -o hello hello.c
./hello

# Use as a drop-in GCC replacement
make CC=/path/to/target/release/ccc
```

For cross-compilation targets install the matching sysroot
(`aarch64-linux-gnu-gcc`, `riscv64-linux-gnu-gcc`).

### GCC-compatible flags

```bash
ccc -S input.c                    # emit assembly
ccc -c input.c                    # compile to object file
ccc -O2 -o output input.c         # optimize (-O0 through -O3, -Os, -Oz)
ccc -g -o output input.c          # DWARF debug info
ccc -DFOO=1 -Iinclude/ input.c    # macros + include paths
ccc -fPIC -shared -o lib.so lib.c # position-independent code
```

Unrecognized flags are silently ignored so `ccc` works as a drop-in in build systems.

### Environment variables

| Variable | Effect |
|----------|--------|
| `CCC_TIME_PASSES` | Print per-pass timing and change counts to stderr |
| `CCC_DISABLE_PASSES` | Disable passes by name (comma-separated, or `all`) |
| `CCC_KEEP_ASM` | Keep intermediate `.s` files next to output |

Pass names: `cfg`, `copyprop`, `narrow`, `simplify`, `constfold`, `gvn`, `licm`,
`ifconv`, `dce`, `ipcp`, `inline`, `ivsr`, `divconst`, `tce`.

---

## Project layout

```
src/
  frontend/     C source → typed AST (preprocessor, lexer, parser, sema)
  ir/           Target-independent SSA IR (lowering, mem2reg)
  passes/       SSA optimization passes (15 passes + shared loop analysis)
  backend/      IR → assembly → ELF (4 architectures)
    live_range.rs   ← LCCC: LiveRange, LinearScanAllocator
    regalloc.rs     ← LCCC: two-pass linear scan activation
  common/       Shared types, symbol table, diagnostics
  driver/       CLI parsing, pipeline orchestration

include/        Bundled C headers (SSE–AVX-512, AES-NI, FMA, SHA, BMI2; NEON)
tests/          Integration tests (main.c + expected output per directory)

lccc-improvements/
  benchmarks/           bench.py runner + 5 C benchmark sources
  register-allocation/  Phase 1 analysis docs (design, integration points)

docs/           Jekyll documentation site source
```

---

## Roadmap

| Phase | Description | Status | Expected gain |
|-------|-------------|--------|---------------|
| 1 | Register allocator analysis & design | ✅ Complete | (prerequisite) |
| 2 | Linear-scan register allocator | ✅ Complete | **+20–25% on reg-pressure code** |
| 3a | Tail-call-to-loop elimination (TCE) | ✅ Complete | **139× on accumulator recursion** |
| 3b | Loop unrolling | Planned | ~1.5–2× on loop-heavy code |
| 4 | SIMD / auto-vectorization (AVX2) | Planned | ~4–8× on FP-heavy code |
| 5 | Profile-guided optimization (PGO) | Planned | ~1.2–1.5× general |

The goal is not to beat GCC — it's to make CCC-compiled programs fast enough for real systems
software, targeting within ~1.5× of GCC on typical workloads.

---

## Testing

```bash
# Unit tests (500 pass)
cargo test --lib

# Integration tests
cargo test --release

# Benchmark suite
python3 lccc-improvements/benchmarks/bench.py --reps 5
```

---

## Licensing

LCCC uses a dual-license model to separate original contributions from CCC-derived code.

**LCCC contributions** (new files, regalloc changes, benchmarks, docs) —
MIT OR Apache-2.0 OR BSD-2-Clause (your choice). See `LICENSE-MIT`, `LICENSE-APACHE`, `LICENSE-BSD`.

**CCC-derived code** (frontend, SSA IR, optimizer, backends, assembler, linker) —
CC0 1.0 Universal (public domain). CCC was released as CC0 by Anthropic.

See [`LICENSING.md`](LICENSING.md) for the full breakdown and per-file guidance.
