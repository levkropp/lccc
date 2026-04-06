#!/usr/bin/env python3
"""
LCCC Comprehensive Benchmark Suite
Compares LCCC vs GCC across performance, correctness, binary size, and compile time.

Usage:
    python3 run_benchmarks.py                    # run all
    python3 run_benchmarks.py --only nbody       # single benchmark
    python3 run_benchmarks.py --reps 10          # more reps
    python3 run_benchmarks.py --skip-perf        # correctness + size only
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
PROGRAMS = HERE / "programs"
REPO_ROOT = HERE.parent.parent
LCCC = REPO_ROOT / "target" / "release" / "lccc"
GCC_INC = subprocess.check_output(
    ["gcc", "-print-file-name=include"], text=True
).strip()

COMPILERS = {
    "LCCC": (str(LCCC), [f"-I{GCC_INC}", "-O2"]),
    "GCC": ("/usr/bin/gcc", ["-O2"]),
}

# Benchmarks: (filename_stem, description, extra_flags, timeout_seconds)
BENCHMARKS = [
    # Original suite
    ("arith_loop", "32-var arithmetic loop (register pressure)", [], 30),
    ("fib", "fib(40) recursive (call overhead)", [], 30),
    ("matmul", "256x256 matrix multiply (FP + cache)", [], 30),
    ("qsort", "Quicksort 1M integers (branching)", [], 30),
    ("sieve", "Sieve of Eratosthenes 10M (memory writes)", [], 30),
    ("tce_sum", "Tail-recursive sum (TCE)", [], 30),
    # New benchmarks
    ("nbody", "N-body simulation (FP-heavy, struct, sqrt)", ["-lm"], 60),
    ("binary_trees", "Binary trees (malloc/free, recursion)", [], 60),
    ("spectral_norm", "Spectral norm (FP dense loops)", ["-lm"], 60),
    ("mandelbrot", "Mandelbrot (FP inner loop, branching)", [], 60),
    ("hash_table", "Hash table (pointer chasing, malloc)", [], 30),
    ("strlen_bench", "String processing (byte ops, memcpy)", [], 30),
    ("switch_dispatch", "Switch dispatch (jump tables)", [], 30),
    ("struct_copy", "Struct copy/field access (ABI, memcpy)", [], 30),
    ("loop_patterns", "Loop patterns (reduce, transform, prefix)", [], 30),
    ("fannkuch", "Fannkuch-Redux (permutations, integer)", [], 120),
    ("ackermann", "Ackermann(3,11) (deep recursion)", [], 60),
    ("bitops", "Bit manipulation (popcount, clz, reverse)", [], 30),
]

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD = "\033[1m"; DIM = "\033[2m"; RED = "\033[31m"; GRN = "\033[32m"
YLW = "\033[33m"; BLU = "\033[34m"; CYN = "\033[36m"; RST = "\033[0m"

def c(code, text):
    return f"{code}{text}{RST}" if sys.stdout.isatty() else text

# ── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class CompileResult:
    ok: bool
    stderr: str = ""
    time_s: float = 0.0
    binary_bytes: int = 0
    text_bytes: int = 0  # .text section size

@dataclass
class RunResult:
    times: list = field(default_factory=list)
    output: str = ""
    returncode: int = 0

@dataclass
class BenchEntry:
    name: str
    desc: str
    compile: dict = field(default_factory=dict)  # compiler -> CompileResult
    run: dict = field(default_factory=dict)       # compiler -> RunResult

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_text_size(binary_path):
    """Get .text section size via objdump or size."""
    try:
        r = subprocess.run(["size", str(binary_path)], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            lines = r.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 1:
                    return int(parts[0])  # .text is first column
    except Exception:
        pass
    return 0

def compile_one(compiler_name, src, out, extra_flags):
    exe, flags = COMPILERS[compiler_name]
    cmd = [exe] + flags + extra_flags + ["-o", str(out), str(src)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    t1 = time.perf_counter()
    if r.returncode != 0:
        return CompileResult(ok=False, stderr=r.stderr.strip(), time_s=t1 - t0)
    sz = out.stat().st_size if out.exists() else 0
    tsz = get_text_size(out)
    return CompileResult(ok=True, time_s=t1 - t0, binary_bytes=sz, text_bytes=tsz)

def run_one(binary, reps, timeout):
    times = []
    output = ""
    rc = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=timeout)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        output = r.stdout.strip()
        rc = r.returncode
    return RunResult(times=times, output=output, returncode=rc)

# ── Main runner ──────────────────────────────────────────────────────────────
def run_benchmarks(only, reps, skip_perf, verbose):
    results = []

    # Find source files - check both programs/ dir and original bench/ dir
    orig_bench = REPO_ROOT / "lccc-improvements" / "benchmarks" / "bench"

    with tempfile.TemporaryDirectory(prefix="lccc_bench_") as tmpdir:
        tmp = Path(tmpdir)

        for bench_stem, desc, extra_flags, timeout in BENCHMARKS:
            if only and bench_stem not in only:
                continue

            # Find source
            src = PROGRAMS / f"{bench_stem}.c"
            if not src.exists():
                # Try original benchmark dir with numbered prefix
                for f in orig_bench.glob(f"*_{bench_stem}.c"):
                    src = f
                    break
                if not src.exists():
                    # Try exact match in orig dir
                    candidate = orig_bench / f"{bench_stem}.c"
                    if candidate.exists():
                        src = candidate

            if not src.exists():
                print(f"  {c(YLW, 'SKIP')} {bench_stem}: source not found")
                continue

            entry = BenchEntry(name=bench_stem, desc=desc)

            print(f"\n{c(BOLD, '►')} {c(BLU, bench_stem)} — {desc}")

            for cname in COMPILERS:
                out = tmp / f"{bench_stem}_{cname}"
                cr = compile_one(cname, src, out, extra_flags)
                entry.compile[cname] = cr

                if not cr.ok:
                    print(f"  {cname:5s} compile: {c(RED, 'FAIL')} ({cr.stderr[:80]})")
                    entry.run[cname] = RunResult()
                    continue

                print(f"  {cname:5s} compile: {cr.time_s:.3f}s  "
                      f"binary={cr.binary_bytes//1024}KB  .text={cr.text_bytes//1024}KB", end="")

                if skip_perf:
                    # Still run once for correctness
                    rr = run_one(out, 1, timeout)
                    entry.run[cname] = rr
                    print(f"  {c(GRN, 'ok') if rr.returncode == 0 else c(RED, 'CRASH')}")
                else:
                    try:
                        rr = run_one(out, reps, timeout)
                        entry.run[cname] = rr
                        best = min(rr.times)
                        avg = mean(rr.times)
                        print(f"  run: {c(GRN, f'{best:.4f}s')} (mean {avg:.4f}s)", end="")
                        if rr.returncode != 0:
                            print(f"  {c(RED, f'exit={rr.returncode}')}", end="")
                        print()
                    except subprocess.TimeoutExpired:
                        entry.run[cname] = RunResult(returncode=-1)
                        print(f"  {c(RED, 'TIMEOUT')}")
                    except Exception as e:
                        entry.run[cname] = RunResult(returncode=-1)
                        print(f"  {c(RED, str(e)[:60])}")

            # Correctness check
            gcc_out = entry.run.get("GCC", RunResult()).output
            lccc_out = entry.run.get("LCCC", RunResult()).output
            if gcc_out and lccc_out:
                if gcc_out == lccc_out:
                    print(f"  correctness: {c(GRN, 'MATCH')}")
                else:
                    print(f"  correctness: {c(RED, 'MISMATCH')}")
                    if verbose:
                        print(f"    GCC:  {gcc_out[:100]}")
                        print(f"    LCCC: {lccc_out[:100]}")
            elif not entry.compile.get("LCCC", CompileResult(ok=False)).ok:
                print(f"  correctness: {c(RED, 'LCCC compile failed')}")

            results.append(entry)

    return results

# ── Report ───────────────────────────────────────────────────────────────────
def print_report(results, reps):
    print(f"\n{'='*100}")
    print(c(BOLD, "  LCCC vs GCC Comprehensive Benchmark Report"))
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  {reps} reps  |  GCC = baseline")
    print(f"{'='*100}\n")

    # Performance table
    print(c(BOLD, "  RUNTIME PERFORMANCE (best-of-N wall seconds)"))
    print(f"  {'Benchmark':<20} {'LCCC':>10} {'GCC':>10} {'Ratio':>10} {'Verdict':>15} {'Correct':>10}")
    print(f"  {'─'*75}")

    perf_ratios = []
    for e in results:
        lccc_r = e.run.get("LCCC", RunResult())
        gcc_r = e.run.get("GCC", RunResult())

        lccc_t = min(lccc_r.times) if lccc_r.times else None
        gcc_t = min(gcc_r.times) if gcc_r.times else None

        correct = "✓" if lccc_r.output and gcc_r.output and lccc_r.output == gcc_r.output else "✗"
        correct_fmt = c(GRN, correct) if correct == "✓" else c(RED, correct)

        if lccc_t is not None and gcc_t is not None and gcc_t > 0:
            ratio = lccc_t / gcc_t
            perf_ratios.append((e.name, ratio))
            if ratio < 0.95:
                verdict = c(GRN, f"{1/ratio:.1f}x faster")
            elif ratio > 1.05:
                verdict = c(RED, f"{ratio:.2f}x slower")
            else:
                verdict = c(DIM, "~parity")
            print(f"  {e.name:<20} {lccc_t:>9.4f}s {gcc_t:>9.4f}s {ratio:>9.2f}x {verdict:>15} {correct_fmt:>10}")
        else:
            lccc_s = f"{lccc_t:.4f}s" if lccc_t else "—"
            gcc_s = f"{gcc_t:.4f}s" if gcc_t else "—"
            print(f"  {e.name:<20} {lccc_s:>10} {gcc_s:>10} {'—':>10} {'—':>15} {correct_fmt:>10}")

    # Binary size table
    print(f"\n{c(BOLD, '  BINARY SIZE')}")
    print(f"  {'Benchmark':<20} {'LCCC':>12} {'GCC':>12} {'Ratio':>10} {'LCCC .text':>12} {'GCC .text':>12} {'.text ratio':>12}")
    print(f"  {'─'*90}")

    size_ratios = []
    for e in results:
        lccc_c = e.compile.get("LCCC", CompileResult(ok=False))
        gcc_c = e.compile.get("GCC", CompileResult(ok=False))
        if lccc_c.ok and gcc_c.ok and gcc_c.binary_bytes > 0:
            ratio = lccc_c.binary_bytes / gcc_c.binary_bytes
            size_ratios.append((e.name, ratio))
            tratio = lccc_c.text_bytes / gcc_c.text_bytes if gcc_c.text_bytes > 0 else 0
            fmt_ratio = c(GRN, f"{ratio:.2f}x") if ratio < 1.1 else c(RED if ratio > 1.5 else YLW, f"{ratio:.2f}x")
            fmt_tratio = c(GRN, f"{tratio:.2f}x") if tratio < 1.1 else c(RED if tratio > 1.5 else YLW, f"{tratio:.2f}x")
            print(f"  {e.name:<20} {lccc_c.binary_bytes:>10,}B {gcc_c.binary_bytes:>10,}B {fmt_ratio:>10} "
                  f"{lccc_c.text_bytes:>10,}B {gcc_c.text_bytes:>10,}B {fmt_tratio:>12}")

    # Compile time table
    print(f"\n{c(BOLD, '  COMPILE TIME')}")
    print(f"  {'Benchmark':<20} {'LCCC':>10} {'GCC':>10} {'Ratio':>10}")
    print(f"  {'─'*50}")

    for e in results:
        lccc_c = e.compile.get("LCCC", CompileResult(ok=False))
        gcc_c = e.compile.get("GCC", CompileResult(ok=False))
        if lccc_c.ok and gcc_c.ok and gcc_c.time_s > 0:
            ratio = lccc_c.time_s / gcc_c.time_s
            fmt = c(GRN, f"{ratio:.2f}x") if ratio < 1.1 else c(RED if ratio > 2 else YLW, f"{ratio:.2f}x")
            print(f"  {e.name:<20} {lccc_c.time_s:>9.3f}s {gcc_c.time_s:>9.3f}s {fmt:>10}")

    # Summary
    print(f"\n{'='*100}")
    print(c(BOLD, "  SUMMARY"))
    print(f"{'='*100}")

    if perf_ratios:
        faster = [(n, r) for n, r in perf_ratios if r < 0.95]
        slower = [(n, r) for n, r in perf_ratios if r > 1.05]
        parity = [(n, r) for n, r in perf_ratios if 0.95 <= r <= 1.05]

        geo_mean = 1.0
        for _, r in perf_ratios:
            geo_mean *= r
        geo_mean = geo_mean ** (1.0 / len(perf_ratios))

        print(f"  Geometric mean LCCC/GCC runtime ratio: {c(BOLD, f'{geo_mean:.3f}x')}")
        print(f"  Faster than GCC: {len(faster)}/{len(perf_ratios)}")
        print(f"  Parity with GCC: {len(parity)}/{len(perf_ratios)}")
        print(f"  Slower than GCC: {len(slower)}/{len(perf_ratios)}")

        if slower:
            print(f"\n  {c(RED, 'Slowest benchmarks (improvement targets):')}:")
            for n, r in sorted(slower, key=lambda x: -x[1]):
                print(f"    {n:<20} {r:.2f}x slower")

        if faster:
            print(f"\n  {c(GRN, 'Fastest benchmarks:')}:")
            for n, r in sorted(faster, key=lambda x: x[1]):
                print(f"    {n:<20} {1/r:.1f}x faster")

    if size_ratios:
        avg_size = mean([r for _, r in size_ratios])
        print(f"\n  Average binary size ratio: {avg_size:.2f}x GCC")

    # Correctness summary
    correct_count = sum(1 for e in results
                       if e.run.get("LCCC", RunResult()).output
                       and e.run.get("GCC", RunResult()).output
                       and e.run.get("LCCC").output == e.run.get("GCC").output)
    compile_fail = sum(1 for e in results
                      if not e.compile.get("LCCC", CompileResult(ok=False)).ok)
    mismatch = len(results) - correct_count - compile_fail
    print(f"\n  Correctness: {correct_count}/{len(results)} match, "
          f"{compile_fail} compile failures, {mismatch} mismatches")

    print()

# ── JSON export ──────────────────────────────────────────────────────────────
def export_json(results, path):
    data = {}
    for e in results:
        entry = {"desc": e.desc, "compilers": {}}
        for cname in COMPILERS:
            cr = e.compile.get(cname, CompileResult(ok=False))
            rr = e.run.get(cname, RunResult())
            entry["compilers"][cname] = {
                "compile_ok": cr.ok,
                "compile_time_s": cr.time_s,
                "binary_bytes": cr.binary_bytes,
                "text_bytes": cr.text_bytes,
                "run_times": rr.times,
                "best_time": min(rr.times) if rr.times else None,
                "output": rr.output,
                "returncode": rr.returncode,
            }
        data[e.name] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON written to: {path}")

# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="LCCC comprehensive benchmark suite")
    p.add_argument("--reps", type=int, default=5, help="repetitions (default 5)")
    p.add_argument("--only", action="append", help="run only these benchmarks")
    p.add_argument("--skip-perf", action="store_true", help="skip performance, do correctness + size only")
    p.add_argument("--json", metavar="FILE", help="export JSON results")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    # Preflight
    if not LCCC.exists():
        print(c(RED, f"ERROR: LCCC not found at {LCCC}"))
        print("  Run: cargo build --release")
        sys.exit(1)

    print(c(BOLD, "\nLCCC Comprehensive Benchmark Suite"))
    print(f"  LCCC: {LCCC}")
    print(f"  GCC:  /usr/bin/gcc")
    print(f"  GCC include: {GCC_INC}")
    print(f"  Reps: {args.reps}")

    results = run_benchmarks(args.only, args.reps, args.skip_perf, args.verbose)
    print_report(results, args.reps)

    if args.json:
        export_json(results, args.json)

if __name__ == "__main__":
    main()
