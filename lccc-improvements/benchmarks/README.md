# LCCC Benchmarking Infrastructure

This directory contains performance testing tools and benchmarks for measuring LCCC improvements.

## Quick Start

```bash
# Build CCC (from project root)
cd /home/min/lccc
cargo build --release

# Run a simple benchmark
echo 'int main() { return 0; }' > /tmp/test.c
time ./target/release/ccc -O0 -o /tmp/test_O0 /tmp/test.c
time ./target/release/ccc -O2 -o /tmp/test_O2 /tmp/test.c
```

## Benchmark Suite

### 1. SQLite Benchmark (Most Important)
The primary benchmark from the CCC vs GCC paper.

```bash
# Download SQLite
wget https://www.sqlite.org/2024/sqlite-amalgamation-3460000.zip
unzip sqlite-amalgamation-3460000.zip
cd sqlite-amalgamation-3460000

# Compile with CCC
time /home/min/lccc/target/release/ccc -O2 -o sqlite3 shell.c sqlite3.c

# Run benchmark (compare runtime)
time ./sqlite3 < benchmark_queries.sql
```

### 2. Micro-benchmarks (Validation)
Small test cases that stress specific optimization areas.

```bash
# Register allocation stress test
gcc -O0 32_vars_switch.c -o test_gcc_O0
time ./test_gcc_O0

/home/min/lccc/target/release/ccc -O0 -o test_ccc_O0 32_vars_switch.c
time ./test_ccc_O0

# Expected: CCC much faster after Phase 1 register allocation fixes
```

### 3. Real-world Projects
- PostgreSQL (237 regression tests)
- Redis (benchmark suite)
- Linux Kernel (defconfig build)

## Measurement Methodology

### Performance Metrics
- **Compilation time**: `time ccc -O2 input.c`
- **Binary size**: `ls -lh output` or `wc -c output`
- **Runtime**: Benchmark queries or microbench iterations
- **Memory usage**: Peak RSS during compilation

### Comparison Method
```bash
# Before optimization
cp target/release/ccc ccc_before

# After optimization
cargo build --release
cp target/release/ccc ccc_after

# Compare
./compare_baseline.sh ccc_before ccc_after test.c
```

## Known Baselines (from CCC vs GCC benchmark)

### SQLite Compilation
| Compiler | Time | Binary | Notes |
|----------|------|--------|-------|
| GCC -O0 | 64.6s | 1.55 MB | Baseline |
| GCC -O2 | 7m23s | 1.40 MB | With optimization |
| CCC -O0 | 87.0s | 4.27 MB | Current CCC |
| CCC -O2 | 87.0s | 4.27 MB | Same as -O0! |

### SQLite Runtime
| Compiler | Time | Speedup | Notes |
|----------|------|---------|-------|
| GCC -O0 | 10.3s | 1x | Baseline |
| GCC -O2 | 6.1s | 1.69x | With optimization |
| CCC | 2h06m | 0.0013x | 737x slower! |

### Per-Query Worst Case
| Query | Type | GCC -O0 | CCC | Slowdown |
|-------|------|---------|-----|----------|
| Q18 | NOT IN subquery | 47ms | 7,432s | **158,129x** |
| Q38 | JOIN + GROUP | 2ms | 52.5s | **26,235x** |
| Q19 | IN subquery | 20ms | 15.5s | **777x** |

## Regression Testing

After each optimization phase, run:

```bash
# Test suite to ensure correctness
cargo test --release

# Validate on real projects
./test_sqlite.sh
./test_postgres.sh

# Performance comparison
./run_sqlite_benchmark.sh --before ./ccc_before --after ./ccc_after
```

## Tools & References

### CCC Bug Reports
Community-maintained fuzzing results: https://github.com/bi6c/ccc-bug-reports
- 70,948 test compilations
- 5 panic signatures
- 5 timeout classes

### Profiling Tools
- `perf` — Linux performance analysis
- `time -v` — Detailed timing
- `valgrind --tool=massif` — Memory profiling
- `gdb` — Debugging (once DWARF fixed)

### Compiler References
- GCC internals: https://gcc.gnu.org/onlinedocs/gccint/
- LLVM optimization: https://llvm.org/docs/Passes/
- Register allocation papers (academic)

## Contributing Benchmarks

When adding a new optimization:

1. **Measure before**: `./run_sqlite_benchmark.sh --before`
2. **Implement optimization**
3. **Measure after**: `./run_sqlite_benchmark.sh --after`
4. **Compare**: Report speedup percentage
5. **Document**: Add results to PR description

Example:
```
### Performance Impact

**Register Allocator Improvement**
- SQLite runtime: 2h 06m → 12.5m (10.1x speedup)
- Binary size: 4.27 MB → 3.94 MB (7.8% reduction)
- Compilation time: 87s → 89s (+2.3%, acceptable)
```

## Next Steps

1. Download SQLite amalgamation
2. Test compilation with CCC
3. Measure baseline runtime
4. Implement register allocator (Phase 1)
5. Re-measure and compare

---

*Benchmarking Infrastructure for LCCC*
