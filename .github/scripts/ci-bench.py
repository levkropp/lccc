#!/usr/bin/env python3
"""
CI benchmark runner for LCCC.
Compiles each benchmark with LCCC and GCC -O2, times them, and reports results.
Portable: auto-detects GCC include path, no hardcoded paths.

Usage:
    python3 ci-bench.py --lccc target/release/lccc --reps 5
    python3 ci-bench.py --lccc target/release/lccc --reps 5 --json results.json
    python3 ci-bench.py --lccc target/release/lccc --reps 5 --summary  # GitHub markdown
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median, stdev

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent.parent
BENCH_DIR = REPO / "lccc-improvements" / "benchmarks" / "bench"

BENCHMARKS = [
    ("01_arith_loop", "Arith Loop"),
    ("02_fib",        "Fibonacci"),
    ("03_matmul",     "MatMul 256²"),
    ("04_qsort",      "Quicksort 1M"),
    ("05_sieve",      "Sieve 10M"),
]


def detect_gcc_inc() -> str:
    r = subprocess.run(
        ["gcc", "-print-file-name=include"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def compile(exe: str, flags: list[str], src: Path, out: Path) -> tuple[bool, str]:
    cmd = [exe, *flags, "-o", str(out), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return False, r.stderr.strip()[:500]
    return True, ""


def time_binary(binary: Path, reps: int) -> tuple[list[float], str]:
    times = []
    output = ""
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
        t1 = time.perf_counter()
        if r.returncode != 0:
            return [], f"exit code {r.returncode}: {r.stderr[:200]}"
        times.append(t1 - t0)
        output = r.stdout.strip()
    return times, output


def run_benchmarks(lccc: str, gcc_inc: str, reps: int):
    results = []

    with tempfile.TemporaryDirectory(prefix="lccc_ci_bench_") as tmpdir:
        tmp = Path(tmpdir)

        for bench_id, bench_name in BENCHMARKS:
            src = BENCH_DIR / f"{bench_id}.c"
            if not src.exists():
                results.append({
                    "id": bench_id, "name": bench_name,
                    "skip": f"source not found: {src}",
                })
                continue

            entry = {"id": bench_id, "name": bench_name}

            for compiler, exe, flags in [
                ("gcc", "gcc", ["-O2"]),
                ("lccc", lccc, [f"-I{gcc_inc}", "-O2"]),
            ]:
                out = tmp / f"{bench_id}_{compiler}"
                ok, err = compile(exe, flags, src, out)
                if not ok:
                    entry[compiler] = {"error": f"compile: {err}"}
                    continue

                times, output = time_binary(out, reps)
                if not times:
                    entry[compiler] = {"error": f"runtime: {output}"}
                    continue

                entry[compiler] = {
                    "times": times,
                    "best": min(times),
                    "mean": mean(times),
                    "median": median(times),
                    "stdev": stdev(times) if len(times) > 1 else 0,
                    "output": output,
                    "binary_bytes": out.stat().st_size,
                }

            # Correctness check
            gcc_out = entry.get("gcc", {}).get("output", "")
            lccc_out = entry.get("lccc", {}).get("output", "")
            if gcc_out and lccc_out:
                entry["correct"] = gcc_out == lccc_out
            else:
                entry["correct"] = None

            results.append(entry)

    return results


def print_terminal(results: list[dict], file=None):
    out = file or sys.stderr
    print(f"\n{'Benchmark':<16} {'GCC -O2':>9} {'LCCC -O2':>10} {'Ratio':>8} {'Correct':>8}", file=out)
    print("-" * 58, file=out)

    for r in results:
        if "skip" in r:
            print(f"  {r['name']:<14} SKIPPED: {r['skip']}", file=out)
            continue

        gcc = r.get("gcc", {})
        lccc = r.get("lccc", {})

        gcc_s = f"{gcc['best']:.3f}s" if "best" in gcc else gcc.get("error", "—")[:12]
        lccc_s = f"{lccc['best']:.3f}s" if "best" in lccc else lccc.get("error", "—")[:12]

        if "best" in gcc and "best" in lccc:
            ratio = lccc["best"] / gcc["best"]
            if ratio < 1.0:
                ratio_s = f"{1/ratio:.0f}x fast"
            else:
                ratio_s = f"{ratio:.2f}x"
        else:
            ratio_s = "—"

        correct_s = "yes" if r.get("correct") else ("MISMATCH" if r.get("correct") is False else "—")

        print(f"  {r['name']:<14} {gcc_s:>9} {lccc_s:>10} {ratio_s:>8} {correct_s:>8}", file=out)

    print(file=out)


def github_summary(results: list[dict], reps: int) -> str:
    lines = [
        "## Benchmark Results: LCCC vs GCC -O2",
        "",
        "| Benchmark | GCC -O2 | LCCC -O2 | LCCC/GCC | Correct |",
        "|-----------|--------:|---------:|---------:|:-------:|",
    ]

    for r in results:
        if "skip" in r:
            lines.append(f"| {r['name']} | — | — | SKIPPED | — |")
            continue

        gcc = r.get("gcc", {})
        lccc = r.get("lccc", {})

        gcc_s = f"{gcc['best']:.3f}s" if "best" in gcc else f"ERR"
        lccc_s = f"{lccc['best']:.3f}s" if "best" in lccc else f"ERR"

        if "best" in gcc and "best" in lccc:
            ratio = lccc["best"] / gcc["best"]
            if ratio < 1.0:
                # LCCC is faster — show as "Nx faster"
                ratio_s = f"**{1/ratio:.0f}x faster**"
            else:
                ratio_s = f"{ratio:.2f}x"
        else:
            ratio_s = "—"

        if r.get("correct") is True:
            correct_s = "pass"
        elif r.get("correct") is False:
            correct_s = "FAIL"
        else:
            correct_s = "—"

        lines.append(f"| {r['name']} | {gcc_s} | {lccc_s} | {ratio_s} | {correct_s} |")

    lines.append("")
    lines.append(f"*Best of {reps} runs, wall-clock time.*")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="LCCC CI benchmark runner")
    p.add_argument("--lccc", required=True, help="Path to LCCC binary")
    p.add_argument("--gcc-inc", default=None, help="GCC include path (auto-detected if omitted)")
    p.add_argument("--reps", type=int, default=5, help="Repetitions per benchmark")
    p.add_argument("--json", metavar="FILE", help="Write JSON results to FILE")
    p.add_argument("--summary", action="store_true", help="Print GitHub Actions markdown summary")
    args = p.parse_args()

    gcc_inc = args.gcc_inc or detect_gcc_inc()

    if not Path(args.lccc).exists():
        print(f"ERROR: LCCC binary not found: {args.lccc}", file=sys.stderr)
        sys.exit(1)

    print(f"LCCC:    {args.lccc}", file=sys.stderr)
    print(f"GCC inc: {gcc_inc}", file=sys.stderr)
    print(f"Reps:    {args.reps}", file=sys.stderr)

    results = run_benchmarks(args.lccc, gcc_inc, args.reps)

    if args.summary:
        print(github_summary(results, args.reps))

    print_terminal(results)

    if args.json:
        # Strip raw output strings for smaller JSON
        export = []
        for r in results:
            entry = {k: v for k, v in r.items()}
            for comp in ("gcc", "lccc"):
                if comp in entry and isinstance(entry[comp], dict):
                    entry[comp] = {k: v for k, v in entry[comp].items() if k != "output"}
            export.append(entry)

        with open(args.json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"JSON written to {args.json}", file=sys.stderr)

    # Fail CI if any correctness mismatch
    mismatches = [r["name"] for r in results if r.get("correct") is False]
    if mismatches:
        print(f"\nERROR: Correctness mismatch in: {', '.join(mismatches)}", file=sys.stderr)
        sys.exit(1)

    # Performance regression checks: verify key claims
    # Fibonacci: rec2iter should make LCCC dramatically faster than GCC
    # We claim 178x; conservatively require at least 10x to account for CI variance
    PERF_THRESHOLDS = {
        "Fibonacci": ("faster", 10.0),  # LCCC must be >= 10x faster than GCC
    }
    perf_failures = []
    for r in results:
        name = r.get("name", "")
        if name not in PERF_THRESHOLDS:
            continue
        direction, threshold = PERF_THRESHOLDS[name]
        gcc = r.get("gcc", {})
        lccc = r.get("lccc", {})
        if "best" not in gcc or "best" not in lccc:
            continue
        ratio = gcc["best"] / lccc["best"]  # >1 means LCCC is faster
        if direction == "faster" and ratio < threshold:
            perf_failures.append(
                f"{name}: expected LCCC >= {threshold:.0f}x faster than GCC, "
                f"got {ratio:.1f}x (LCCC={lccc['best']:.4f}s, GCC={gcc['best']:.4f}s)"
            )
    if perf_failures:
        print(f"\nERROR: Performance regression detected:", file=sys.stderr)
        for f in perf_failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
