#!/usr/bin/env bash
# Verify Fibonacci performance claim: LCCC should be dramatically faster
# than GCC on recursive Fibonacci thanks to the rec2iter optimization.
#
# Usage: ./tests/bench_fib_verify.sh [path-to-lccc] [reps]
set -euo pipefail

LCCC="${1:-target/release/lccc}"
REPS="${2:-5}"
GCC_INC="$(gcc -print-file-name=include)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/../lccc-improvements/benchmarks/bench/02_fib.c"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== Fibonacci Performance Verification ==="
echo "LCCC: $LCCC"
echo "Reps: $REPS"
echo ""

# 1. Compile with both compilers
echo "Compiling..."
gcc -O2 -o "$TMPDIR/fib_gcc" "$SRC"
"$LCCC" -I"$GCC_INC" -O2 -o "$TMPDIR/fib_lccc" "$SRC"

# 2. Verify correctness
GCC_OUT=$("$TMPDIR/fib_gcc")
LCCC_OUT=$("$TMPDIR/fib_lccc")
if [ "$GCC_OUT" != "$LCCC_OUT" ]; then
    echo "FAIL: Output mismatch"
    echo "  GCC:  $GCC_OUT"
    echo "  LCCC: $LCCC_OUT"
    exit 1
fi
echo "Correctness: PASS ($GCC_OUT)"

# 3. Verify rec2iter fires
REC_OUT=$(LCCC_DEBUG_RECURSION=1 "$LCCC" -I"$GCC_INC" -O2 -o /dev/null "$SRC" 2>&1 || true)
if echo "$REC_OUT" | grep -q "REC→ITER.*Transformed"; then
    echo "rec2iter pass: ACTIVE"
else
    echo "FAIL: rec2iter pass did not fire!"
    echo "  Debug output: $REC_OUT"
    exit 1
fi

# 4. Benchmark
echo ""
echo "Benchmarking ($REPS runs each)..."

best_gcc=999
best_lccc=999
for i in $(seq 1 "$REPS"); do
    t=$( { time "$TMPDIR/fib_gcc" > /dev/null; } 2>&1 | grep real | sed 's/real\t//;s/m/*60+/;s/s//' | bc -l )
    if (( $(echo "$t < $best_gcc" | bc -l) )); then best_gcc=$t; fi
done

for i in $(seq 1 "$REPS"); do
    t=$( { time "$TMPDIR/fib_lccc" > /dev/null; } 2>&1 | grep real | sed 's/real\t//;s/m/*60+/;s/s//' | bc -l )
    if (( $(echo "$t < $best_lccc" | bc -l) )); then best_lccc=$t; fi
done

ratio=$(echo "$best_gcc / $best_lccc" | bc -l)
printf "GCC  best: %.4fs\n" "$best_gcc"
printf "LCCC best: %.4fs\n" "$best_lccc"
printf "Speedup:   %.0fx faster\n" "$ratio"

# 5. Assert minimum speedup (conservative: 10x, we typically see 100-400x)
MIN_SPEEDUP=10
if (( $(echo "$ratio < $MIN_SPEEDUP" | bc -l) )); then
    echo ""
    echo "FAIL: Expected at least ${MIN_SPEEDUP}x speedup, got ${ratio}x"
    echo "The rec2iter optimization may not be working correctly."
    exit 1
fi

echo ""
echo "PASS: LCCC is $(printf '%.0f' "$ratio")x faster than GCC on recursive Fibonacci"

# 6. Run the comprehensive correctness test
echo ""
echo "Running comprehensive rec2iter correctness test..."
"$LCCC" -I"$GCC_INC" -O2 -o "$TMPDIR/fib_comprehensive" "$SCRIPT_DIR/fib_rec2iter.c"
COMP_OUT=$("$TMPDIR/fib_comprehensive")
if [ "$COMP_OUT" = "ALL PASS" ]; then
    echo "Comprehensive test: PASS (includes fib(90) — impossible without rec2iter)"
else
    echo "FAIL: Comprehensive test output: $COMP_OUT"
    exit 1
fi
