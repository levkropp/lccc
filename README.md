# LCCC — Lev's Claude's C Compiler

> An optimized fork of [CCC](https://github.com/anthropics/claudes-c-compiler) with a two-pass
> linear-scan register allocator, phi-copy stack coalescing, loop unrolling, FP intrinsic
> lowering, FP peephole optimization, and AVX2 auto-vectorization with remainder loops. **+42% faster** on register-pressure code, **~2× of GCC** on matrix multiply (was 6.0×).

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
two-pass linear-scan allocator (Poletto & Sarkar 1999). Phase 3 adds tail-call elimination and
phi-copy stack slot coalescing. Phase 4 adds loop unrolling and FP intrinsic lowering. Phase 5 adds
FP peephole optimization that eliminates GPR↔XMM round-trips and stack spills. Phase 6 adds SSE2
auto-vectorization (2-wide) for matmul-style loops. Phase 7a upgrades to AVX2 vectorization (4-wide),
and Phase 7b implements remainder loops for production-ready vectorization at any array size.
Together these yield +42% speedup on register-pressure code and bring the matmul GCC gap from 6.0× to ~2×,
while keeping all 514 tests green.

```
C source
  │  frontend: lex → parse → sema → IR lowering
  ▼
SSA IR
  │  optimizer: TCE · loop-unroll · vectorize(AVX2) · GVN · LICM · IPCP · DCE · const-fold · inline
  ▼
Optimized IR
  │  regalloc (LCCC): two-pass linear scan over live intervals
  │    pass 1: callee-saved  ↔ all eligible values
  │    pass 2: caller-saved  ↔ non-call-spanning unallocated values
  ▼
Machine code  (x86-64 · AArch64 · RISC-V 64 · i686)
  │  peephole: FP round-trip elim · memory fold · spill elim · rcx copy fold
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
| `arith_loop` — 32-var arithmetic, 10 M iters | **0.103 s** | 0.146 s | 0.068 s | **+42% faster** | 1.50× slower |
| `sieve` — primes to 10 M | **0.036 s** | 0.045 s | 0.024 s | **+25% faster** | 1.50× slower |
| `qsort` — sort 1 M integers | 0.096 s | 0.095 s | 0.087 s | ≈ equal | 1.10× slower |
| `fib(40)` — recursive Fibonacci | 0.352 s | 0.354 s | 0.096 s | ≈ equal | 3.68× slower |
| `matmul` — 256×256 double | **0.008 s** | 0.029 s | 0.004 s | **+263% faster** | ~2.0× slower |
| `tce_sum` — tail-recursive sum(10M) | **0.008 s** | 1.09 s | 0.008 s | **139× faster** | ≈ equal |

The `arith_loop` gain comes from linear-scan register allocation + phi-copy stack coalescing
(eliminates ~20 redundant stack-to-stack copies per iteration).
The `sieve` gain comes from linear scan keeping the inner-loop counter in a register + loop
unrolling the prime-counting pass.
The `matmul` gain comes from Phase 4 FP intrinsic lowering + Phase 5 FP peephole optimization
(eliminates GPR↔XMM round-trips, folds memory operands, removes stack spills — 33→20 inner loop instructions)
+ Phase 6 SSE2 auto-vectorization (2-wide, ~2× speedup) + Phase 7 AVX2 upgrade (4-wide, ~2× additional speedup).
The remaining `matmul` gap is GCC's more aggressive loop optimizations (unroll-and-jam, strength reduction).
The `tce_sum` gain comes from tail-call elimination converting 10M recursive calls into a loop.

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

## Phi-copy stack slot coalescing

When CCC's phi elimination lowers SSA phi nodes to Copy instructions, it creates a
"double-slot" problem for spilled loop variables: both the phi destination (`%i`) and
its backedge update (`%i_next`) get separate stack slots, with the loop body copying
one to the other on every iteration.

For a 32-variable loop at `-O2`, this produces ~20 redundant stack-to-stack copies per
iteration (`movq mem, %rax; movq %rax, mem`), adding measurable overhead.

**Root cause:** standard copy coalescing refused to merge these because `%i` is
multi-defined (phi elimination creates one `Copy { dest: %i, src: ... }` per predecessor
block), and `%i_next` is used in a different block than where the copy appears.

**Fix:** For the phi-copy pattern — where `%i_next` is defined and consumed in the same
backedge block (sole use = the copy), but `%i` is used cross-block — alias in the
*reversed* direction: `%i_next` borrows `%i`'s slot. Since `%i_next`'s only use is the
copy, and `%i`'s slot is already live across all of `%i`'s uses (including other blocks),
this is safe. The backedge copy becomes a same-slot no-op and is dropped by the code
generator (`generate_copy` checks `dest_slot == src_slot`).

Result on `arith_loop` (32 spilled variables): 0.124 s → 0.104 s (**+19% additional speedup**).

The analysis lives in [`src/backend/stack_layout/copy_coalescing.rs`](src/backend/stack_layout/copy_coalescing.rs).

---

## Loop unrolling

Small inner loops are unrolled 2×–8× at compile time. The pass uses an **early-exit** strategy
that handles non-multiple trip counts without a separate cleanup loop:

```
header:  %iv = phi [init, %iv_next]
         %cond = cmp %iv, limit
         branch %cond → exit / body

[original body blocks] → exit_check_1

exit_check_1:  %iv_1 = %iv + step
               branch cmp(%iv_1, limit) → exit / body_copy_2

[body_copy_2 — all value IDs renamed] → exit_check_2
...
exit_check_{K-1} → [body_copy_K] → latch

latch: %iv_next = %iv_{K-1} + step   ← was %iv + step
       branch header
```

Each exit check fires as soon as the IV exceeds the limit, so the unrolled loop is correct
for any trip count — no remainder loop needed.

**Eligibility:** single latch, preheader exists, ≤8 body-work blocks, constant IV step,
detectable exit condition (Cmp with loop-invariant limit), no calls/atomics in body.

**Unroll factors:** 8× for ≤8 body instructions, 4× for 9–20, 2× for 21–60, skip above 60.

The pass lives in [`src/passes/loop_unroll.rs`](src/passes/loop_unroll.rs) and runs once at
`iter=0` before GVN/LICM so that subsequent passes can optimize the unrolled copies.

---

## SSE2 Auto-vectorization

Innermost loops with matmul-style accumulation patterns are automatically vectorized to process
2 doubles per iteration using SSE2 packed instructions.

**Pattern recognized:**
```c
for (int j = 0; j < N; j++)
    C[i][j] += A[i][k] * B[k][j];  // scalar accumulation
```

**Transformed to:**
```c
for (int j = 0; j < N/2; j++)  // Half the iterations
    FmaF64x2(&C[i][j*2], &A[i][k], &B[k][j*2]);  // Processes elements j*2 and j*2+1
```

**Generated assembly:**
```asm
movsd   (%rcx), %xmm1      # Load scalar A[i][k]
unpcklpd %xmm1, %xmm1      # Broadcast to both lanes
movupd  (%rdx), %xmm0      # Load 2 doubles from B
mulpd   %xmm1, %xmm0       # Packed multiply (2 ops)
addpd   (%rax), %xmm0      # Packed add with C (2 ops)
movupd  %xmm0, (%rax)      # Store 2 results
```

**How it works:**

1. **IV tracking** — Traces the induction variable through casts/copies, working backward from GEP instructions to find the actual j-loop IV
2. **Loop bound** — Modifies comparisons from `j < N` to `j < N/2` (inserts udiv for dynamic N)
3. **Array indexing** — Inserts multiply instructions to change GEP offsets from `j` to `j*2`, ensuring stride-16 addressing
4. **Intrinsic insertion** — Replaces scalar load/mul/add/store with `FmaF64x2` intrinsic
5. **Code generation** — Backend emits SSE2 packed instructions (movupd, mulpd, addpd, unpcklpd)

The pass runs at `iter=0` (early) before other optimizations and correctly handles strength-reduced
loops. It processes 2 elements per iteration while keeping the IV incrementing by 1 (backend-friendly).

**Remainder loops (Phase 7b):** Automatically inserted to handle N % vec_width != 0:
- AVX2 (4-wide): Handles remainders 1–3 with scalar loop
- SSE2 (2-wide): Handles remainder 1 with scalar loop
- Example for N=255 with AVX2: Vectorized loop processes 63 iterations (indices 0–251), remainder loop handles 3 scalar iterations (indices 252–254)
- Zero overhead when N is divisible by vector width; <3% overhead on average for non-divisible N

**Limitations:** Only matmul-style patterns supported (load, multiply, add, store).

The pass lives in [`src/passes/vectorize.rs`](src/passes/vectorize.rs) (1400+ lines).

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
| `LCCC_DEBUG_VECTORIZE` | Print vectorization pattern matching and transformation details |

Pass names: `cfg`, `copyprop`, `narrow`, `simplify`, `constfold`, `gvn`, `licm`,
`ifconv`, `dce`, `ipcp`, `inline`, `ivsr`, `divconst`, `tce`, `unroll`, `vectorize`.

---

## Project layout

```
src/
  frontend/     C source → typed AST (preprocessor, lexer, parser, sema)
  ir/           Target-independent SSA IR (lowering, mem2reg)
  passes/       SSA optimization passes (16 passes + shared loop analysis)
  backend/      IR → assembly → ELF (4 architectures)
    live_range.rs   ← LCCC: LiveRange, LinearScanAllocator
    regalloc.rs     ← LCCC: two-pass linear scan activation
  common/       Shared types, symbol table, diagnostics
  driver/       CLI parsing, pipeline orchestration

include/        Bundled C headers (SSE–AVX-512, AES-NI, FMA, SHA, BMI2; NEON)
tests/          Integration tests (main.c + expected output per directory)

lccc-improvements/
  benchmarks/           bench.py runner + 6 C benchmark sources
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
| 3b | Phi-copy stack slot coalescing | ✅ Complete | **+additional 20% on loop-heavy code** |
| 4 | Loop unrolling + FP intrinsic lowering | ✅ Complete | **+45% matmul vs CCC; sieve counting loop 8×** |
| 5 | FP peephole optimization | ✅ Complete | **+additional 41% matmul vs CCC** |
| 6 | SSE2 auto-vectorization (2-wide) | ✅ Complete | **~2× on matmul-style FP loops** |
| 7a | AVX2 vectorization (4-wide) | ✅ Complete | **~2× additional on matmul vs SSE2** |
| 7b | Remainder loop handling | ✅ Complete | **Production-ready vectorization for any N** |
| 8 | Better function inlining | Planned | ~1.8× on fib(40) |
| 9 | Loop strength reduction | Planned | Eliminate redundant addressing |
| 10 | Profile-guided optimization (PGO) | Planned | ~1.2–1.5× general |

The goal is not to beat GCC — it's to make CCC-compiled programs fast enough for real systems
software, targeting within ~1.5× of GCC on typical workloads.

---

## Testing

```bash
# Unit tests (508 pass)
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
