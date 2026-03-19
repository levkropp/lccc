# LCCC Benchmark Results — Phase 2 Status & Performance Report

**Date:** 2026-03-19
**LCCC commit:** `33ebb7bc` (Phase 2: Implement core linear scan register allocator, 2a-2d complete)
**CCC commit:** `6f1b99ac` (upstream, without linear scan)
**GCC version:** 15.2.0 (Ubuntu 15.2.0-4ubuntu4)
**Host:** x86-64 Linux, compiled with `-O2`

---

## Phase 2 Status

| Sub-phase | Description | Status |
|-----------|-------------|--------|
| Phase 2a | LiveRange / ActiveInterval / LinearScanAllocator data structures | ✓ Complete |
| Phase 2b | `build_live_ranges()`, `collect_uses()`, register hint detection | ✓ Complete |
| Phase 2c | Core algorithm: expire, find_free, allocate_range, spill, run() | ✓ Complete |
| Phase 2d | Integration into regalloc.rs + 7-test suite (all pass) | ✓ Complete (code present) |
| **Activation** | Replace `allocate_registers()` body with linear scan call | ✗ NOT done |

**Summary:** The `LinearScanAllocator` is fully implemented in `ccc/src/backend/live_range.rs`
(796 lines). A wrapper `allocate_registers_linear_scan()` exists in `regalloc.rs` and all
7 unit tests pass. However, this function is marked `#[allow(dead_code)]` and the existing
three-phase allocator is still the default. Phase 2 code is written but not activated.
Week 3 work (enable + validate + benchmark) is outstanding.

---

## Performance Results

### Timing (wall clock, `time` command)

| Benchmark | LCCC | CCC | GCC -O2 | LCCC/GCC | CCC/GCC |
|-----------|------|-----|---------|----------|---------|
| **01 Arith Loop** (32 vars, 10M iters) | 0.160s | 0.160s | 0.069s | **2.32x slower** | 2.32x slower |
| **02 Fibonacci** fib(40) recursive | 0.361s | 0.360s | 0.097s | **3.72x slower** | 3.71x slower |
| **03 Matrix Mul** 256×256 doubles | 0.029s | 0.029s | 0.004s | **7.25x slower** | 7.25x slower |
| **04 Quicksort** 1M integers | 0.096s | 0.096s | 0.088s | **1.09x slower** | 1.09x slower |
| **05 Sieve** primes up to 10M | 0.047s | 0.046s | 0.025s | **1.88x slower** | 1.84x slower |

### Binary Size

| Benchmark | LCCC | CCC | GCC |
|-----------|------|-----|-----|
| 01 Arith Loop | 14,520 B | 14,520 B | 16,080 B |
| 02 Fibonacci | 14,520 B | 14,520 B | 16,064 B |
| 03 Matrix Mul | 14,520 B | 14,520 B | 16,152 B |
| 04 Quicksort | 14,528 B | 14,528 B | 16,136 B |
| 05 Sieve | 14,528 B | 14,528 B | 16,152 B |

LCCC/CCC produce **~12% smaller binaries** than GCC (surprising — GCC's CET/endbr64
instrumentation and alignment padding account for most of this).

### Code Quality: Arith Loop Disassembly (register allocation)

| Metric | LCCC | CCC | GCC |
|--------|------|-----|-----|
| Stack frame size | 0x1f0 (496 bytes) | 0x1f0 (496 bytes) | ~0x60 (96 bytes) |
| rbp/rsp memory refs in binary | 298 | 298 | 110 |
| Memory ops vs register ratio | ~2.7x more than GCC | ~2.7x more than GCC | baseline |

GCC keeps most of the 32 variables in registers (r8-r15, rbx, rbp, rsi, rdi, rcx, rdx).
LCCC/CCC spills most to a 496-byte stack frame with load-modify-store sequences.

---

## Key Findings

### LCCC vs CCC
**Effectively identical.** All benchmark results are within noise (0.001s or less difference).
This is expected: the linear scan allocator exists in the LCCC submodule but is not
activated. The two compilers produce byte-for-byte identical object files for these benchmarks.

### LCCC/CCC vs GCC -O2
- **Arithmetic-heavy code (01, 02):** 2.3–3.7x slower — register pressure causes excessive
  stack spilling; this is exactly the bottleneck Phase 2 targets.
- **Floating-point/memory (03):** 7.25x slower — matrix multiply is dominated by
  FP throughput; CCC's scalar code vs GCC's auto-vectorized inner loop.
- **Library-dominated (04):** Only 9% slower — `qsort()` is a libc call; compiler quality
  matters less here.
- **Memory-bound (05):** 1.88x slower — sieve is memory-bound; moderate register pressure.

### Expected Impact of Enabling Linear Scan (Week 3)
Based on disassembly, benchmark 01 (arith loop) should see the biggest improvement from
Phase 2 activation. GCC's 2.32x advantage over LCCC is consistent with the 3–4x target
documented in `LINEAR_SCAN_DESIGN.md`. The FP/vectorization gap (benchmark 03) requires
Phase 3+ work (SIMD, vectorization).

---

## How to Reproduce

```bash
LCCC=/home/min/lccc/target/release/ccc
CCC=/home/min/ccc-upstream/target/release/ccc   # fresh clone of upstream
GCC=/usr/bin/gcc
GCC_INC="-I/usr/lib/gcc/x86_64-linux-gnu/15/include"
BENCH=/home/min/lccc/lccc-improvements/benchmarks/bench

# Compile
for bench in 01_arith_loop 02_fib 03_matmul 04_qsort 05_sieve; do
  $LCCC $GCC_INC -O2 -o /tmp/${bench}_lccc $BENCH/${bench}.c
  $CCC  $GCC_INC -O2 -o /tmp/${bench}_ccc  $BENCH/${bench}.c
  $GCC  -O2 -o /tmp/${bench}_gcc           $BENCH/${bench}.c
done

# Time
time /tmp/${bench}_lccc
time /tmp/${bench}_ccc
time /tmp/${bench}_gcc
```

---

## Next Steps (Week 3)

1. **Activate linear scan allocator** — replace body of `allocate_registers()` in
   `regalloc.rs` with call to `allocate_registers_linear_scan()`
2. **Re-run these benchmarks** — measure actual improvement (expected: 2–4x on bench 01/02)
3. **Run CCC test suite** — validate correctness on all existing tests
4. **Run SQLite/PostgreSQL** — real-world validation before declaring success
