#!/usr/bin/env python3
"""
LCCC Benchmark Suite
Compares LCCC (linear-scan allocator), CCC (upstream), and GCC -O2.
Generates a rich terminal report and a Markdown file.

Usage:
    python3 bench.py              # run all benchmarks, 5 reps each
    python3 bench.py --reps 10   # more reps
    python3 bench.py --bench 01_arith_loop  # single benchmark
    python3 bench.py --md report.md         # write Markdown report
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent.resolve()
BENCH_DIR  = HERE / "bench"
REPO_ROOT  = HERE.parent.parent          # /home/min/lccc
LCCC       = REPO_ROOT / "target/release/ccc"
CCC        = Path("/home/min/ccc-upstream/target/release/ccc")
GCC        = Path("/usr/bin/gcc")
GCC_INC    = "/usr/lib/gcc/x86_64-linux-gnu/15/include"

COMPILERS = {
    "LCCC":  (str(LCCC),  [f"-I{GCC_INC}", "-O2"]),
    "CCC":   (str(CCC),   [f"-I{GCC_INC}", "-O2"]),
    "GCC":   (str(GCC),   ["-O2"]),
}

BENCHMARKS = [
    ("01_arith_loop", "Arith Loop",    "32-var arithmetic, 10M iters  (register pressure)"),
    ("02_fib",        "Fibonacci",     "fib(40) recursive              (call overhead)"),
    ("03_matmul",     "MatMul 256²",   "256×256 double matrix multiply (FP + cache)"),
    ("04_qsort",      "Quicksort 1M",  "sort 1M integers               (branching)"),
    ("05_sieve",      "Sieve 10M",     "primes to 10M                  (memory writes)"),
]

# ── Terminal colours ───────────────────────────────────────────────────────────
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[31m"
GRN   = "\033[32m"
YLW   = "\033[33m"
BLU   = "\033[34m"
MAG   = "\033[35m"
CYN   = "\033[36m"
RST   = "\033[0m"

def has_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def c(code: str, text: str) -> str:
    return f"{code}{text}{RST}" if has_color() else text

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    times: list[float]           # wall-clock seconds per repetition
    binary_bytes: int
    output: str                  # stdout of the binary (for correctness check)
    compile_ok: bool = True
    compile_err: str = ""

    @property
    def min_s(self)    -> float: return min(self.times) if self.times else float("nan")
    @property
    def mean_s(self)   -> float: return mean(self.times) if self.times else float("nan")
    @property
    def median_s(self) -> float: return median(self.times) if self.times else float("nan")
    @property
    def stdev_s(self)  -> Optional[float]:
        return stdev(self.times) if len(self.times) > 1 else None

@dataclass
class BenchResult:
    bench_id: str
    bench_name: str
    runs: dict[str, RunResult] = field(default_factory=dict)   # compiler → RunResult

# ── Compilation ────────────────────────────────────────────────────────────────
def compile_benchmark(compiler: str, src: Path, out: Path) -> tuple[bool, str]:
    exe, flags = COMPILERS[compiler]
    cmd = [exe, *flags, "-o", str(out), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return False, r.stderr.strip()
    return True, ""

# ── Timing ─────────────────────────────────────────────────────────────────────
def time_binary(binary: Path, reps: int) -> tuple[list[float], str]:
    """Return (list-of-elapsed-seconds, stdout-of-last-run)."""
    times = []
    output = ""
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
        t1 = time.perf_counter()
        if r.returncode != 0:
            raise RuntimeError(f"Binary exited with {r.returncode}: {r.stderr[:200]}")
        times.append(t1 - t0)
        output = r.stdout.strip()
    return times, output

# ── Binary size ────────────────────────────────────────────────────────────────
def binary_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0

# ── Main benchmark runner ──────────────────────────────────────────────────────
def run_all(bench_ids: list[str], reps: int, verbose: bool) -> list[BenchResult]:
    results: list[BenchResult] = []

    with tempfile.TemporaryDirectory(prefix="lccc_bench_") as tmpdir:
        tmp = Path(tmpdir)

        for bench_id, bench_name, _ in BENCHMARKS:
            if bench_ids and bench_id not in bench_ids:
                continue
            src = BENCH_DIR / f"{bench_id}.c"
            if not src.exists():
                print(f"  {c(YLW,'SKIP')} {bench_id}: source not found ({src})")
                continue

            br = BenchResult(bench_id=bench_id, bench_name=bench_name)
            print(f"\n{c(BOLD,'►')} {c(BLU, bench_name)} — {reps} reps")

            for cname in COMPILERS:
                out = tmp / f"{bench_id}_{cname}"
                print(f"  {c(DIM, cname+':')}", end=" ", flush=True)

                ok, err = compile_benchmark(cname, src, out)
                if not ok:
                    br.runs[cname] = RunResult(
                        times=[], binary_bytes=0, output="",
                        compile_ok=False, compile_err=err,
                    )
                    print(c(RED, "COMPILE FAILED"))
                    if verbose:
                        print(f"    {err}")
                    continue

                bsize = binary_size(out)
                try:
                    times, output = time_binary(out, reps)
                except RuntimeError as e:
                    br.runs[cname] = RunResult(
                        times=[], binary_bytes=bsize, output="",
                        compile_ok=True, compile_err=str(e),
                    )
                    print(c(RED, f"RUNTIME ERROR: {e}"))
                    continue

                br.runs[cname] = RunResult(
                    times=times, binary_bytes=bsize, output=output,
                )
                best = min(times)
                print(f"{c(GRN, f'{best:.3f}s')}  "
                      f"(mean {mean(times):.3f}s, {bsize//1024}KB)")

            results.append(br)

    return results

# ── Formatting helpers ─────────────────────────────────────────────────────────
def fmt_ratio(ratio: float) -> str:
    """Format a speed ratio: green if faster, red if slower, grey if ~equal."""
    if ratio < 0.97:
        return c(GRN, f"{ratio:.2f}×")
    elif ratio > 1.03:
        return c(RED, f"{ratio:.2f}×")
    else:
        return c(DIM, f"{ratio:.2f}×")

def speedup_label(ratio: float) -> str:
    """Label a vs-GCC ratio for the report."""
    if ratio <= 1.0:
        return f"{1/ratio:.1f}× faster than GCC"
    else:
        return f"{ratio:.2f}× slower than GCC"

def bar(fraction: float, width: int = 20) -> str:
    """Simple ASCII bar for the terminal."""
    filled = round(fraction * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)

# ── Terminal report ────────────────────────────────────────────────────────────
def print_report(results: list[BenchResult]) -> None:
    sep = "─" * 82

    print(f"\n{c(BOLD, sep)}")
    print(c(BOLD, "  LCCC Benchmark Report"))
    print(c(DIM,  f"  {time.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(c(BOLD, sep))

    # Header
    print(f"\n{'Benchmark':<22} {'Compiler':<8} {'Best':>8} {'Mean':>8} {'±':>7} "
          f"{'vs GCC':>9}  {'Size':>6}  Correctness")
    print("─" * 82)

    for br in results:
        gcc_best = br.runs.get("GCC", RunResult([], 0, "")).min_s
        gcc_out  = br.runs.get("GCC", RunResult([], 0, "")).output

        for i, cname in enumerate(COMPILERS):
            rr = br.runs.get(cname)
            name_col = br.bench_name if i == 0 else ""

            if rr is None or not rr.compile_ok:
                err = (rr.compile_err[:30] if rr else "not run")
                print(f"  {name_col:<20} {cname:<8} {'—':>8} {'—':>8} {'—':>7} "
                      f"{'—':>9}  {'—':>6}  {c(RED, 'COMPILE FAIL')}")
                continue
            if not rr.times:
                print(f"  {name_col:<20} {cname:<8} {'—':>8} {'—':>8} {'—':>7} "
                      f"{'—':>9}  {'—':>6}  {c(RED, 'RUNTIME ERR')}")
                continue

            best  = rr.min_s
            avg   = rr.mean_s
            sd    = rr.stdev_s
            ratio = best / gcc_best if gcc_best and gcc_best > 0 else float("nan")
            correct = "✓" if rr.output == gcc_out else c(RED, "✗ MISMATCH")

            sd_str  = f"±{sd:.3f}" if sd is not None else "    —"
            sz_str  = f"{rr.binary_bytes//1024}KB"

            if cname == "GCC":
                ratio_str = c(DIM, "baseline")
            else:
                ratio_str = fmt_ratio(ratio)

            print(f"  {name_col:<20} {cname:<8} {best:>7.3f}s {avg:>7.3f}s {sd_str:>7}  "
                  f"{ratio_str:>9}  {sz_str:>6}  {correct}")

        print()

    # Summary table
    print(c(BOLD, sep))
    print(c(BOLD, "  Summary: LCCC vs GCC  (best-of-N wall time)"))
    print(c(BOLD, sep))
    print(f"  {'Benchmark':<22} {'LCCC/GCC':>10}  {'CCC/GCC':>10}  {'LCCC/CCC':>10}  Bar (LCCC vs GCC)")
    print("  " + "─" * 78)

    for br in results:
        gcc  = br.runs.get("GCC")
        lccc = br.runs.get("LCCC")
        ccc  = br.runs.get("CCC")

        def safe_ratio(a, b):
            if a and b and a.times and b.times and b.min_s > 0:
                return a.min_s / b.min_s
            return None

        lg = safe_ratio(lccc, gcc)
        cg = safe_ratio(ccc, gcc)
        lc = safe_ratio(lccc, ccc)

        def fmt(r):
            return fmt_ratio(r) if r is not None else c(DIM, "     —")

        bar_width = 16
        bar_fill  = min(1.0, (lg or 1.0))
        bar_str   = bar(bar_fill, bar_width)

        print(f"  {br.bench_name:<22} {fmt(lg):>10}  {fmt(cg):>10}  {fmt(lc):>10}  {bar_str}")

    print()

# ── Markdown report ────────────────────────────────────────────────────────────
def write_markdown(results: list[BenchResult], path: str) -> None:
    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        f"# LCCC Benchmark Report",
        f"",
        f"Generated: {ts}",
        f"",
        f"## Compilers",
        f"",
        f"| Name | Binary |",
        f"|------|--------|",
        f"| LCCC | `{LCCC}` (linear-scan allocator active) |",
        f"| CCC  | `{CCC}` (upstream, three-phase allocator) |",
        f"| GCC  | `{GCC}` (`-O2`) |",
        f"",
        f"## Results",
        f"",
        f"All times are **best-of-N wall-clock seconds** (lower is better).  ",
        f"Ratios are relative to GCC -O2.",
        f"",
        f"| Benchmark | Description | LCCC | CCC | GCC | LCCC/GCC | CCC/GCC | Size LCCC | Size GCC |",
        f"|-----------|-------------|-----:|----:|----:|---------:|--------:|----------:|---------:|",
    ]

    for br in results:
        desc = next((d for bid, _, d in BENCHMARKS if bid == br.bench_id), "")
        gcc  = br.runs.get("GCC")
        lccc = br.runs.get("LCCC")
        ccc  = br.runs.get("CCC")

        def t(rr): return f"{rr.min_s:.3f}s" if rr and rr.times else "—"
        def sz(rr): return f"{rr.binary_bytes//1024} KB" if rr and rr.binary_bytes else "—"
        def ratio(a, b):
            if a and b and a.times and b.times and b.min_s > 0:
                return f"{a.min_s/b.min_s:.2f}×"
            return "—"

        lines.append(
            f"| **{br.bench_name}** | {desc} | {t(lccc)} | {t(ccc)} | {t(gcc)} "
            f"| {ratio(lccc,gcc)} | {ratio(ccc,gcc)} | {sz(lccc)} | {sz(gcc)} |"
        )

    lines += [
        f"",
        f"## Correctness",
        f"",
        f"| Benchmark | LCCC == GCC | CCC == GCC |",
        f"|-----------|:-----------:|:----------:|",
    ]

    for br in results:
        gcc  = br.runs.get("GCC")
        lccc = br.runs.get("LCCC")
        ccc  = br.runs.get("CCC")
        def ok(a, b):
            if a and b and a.output and b.output:
                return "✅" if a.output == b.output else "❌"
            return "—"
        lines.append(f"| {br.bench_name} | {ok(lccc, gcc)} | {ok(ccc, gcc)} |")

    lines += [
        f"",
        f"## Notes",
        f"",
        f"- LCCC uses the new linear-scan register allocator (Phase 2 of the LCCC",
        f"  optimization roadmap). The allocator runs two passes:",
        f"  1. **Callee-saved pass**: linear scan over all eligible IR values,",
        f"     assigning callee-saved registers (safe across calls).",
        f"  2. **Caller-saved pass**: linear scan over unallocated, non-call-spanning",
        f"     values, assigning caller-saved registers.",
        f"- CCC uses the original three-phase greedy allocator.",
        f"- GCC `-O2` is the performance baseline.",
        f"",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Markdown report written to: {path}")

# ── JSON export ────────────────────────────────────────────────────────────────
def export_json(results: list[BenchResult], path: str) -> None:
    data = {}
    for br in results:
        data[br.bench_id] = {}
        for cname, rr in br.runs.items():
            data[br.bench_id][cname] = {
                "times": rr.times,
                "min_s": rr.min_s,
                "mean_s": rr.mean_s,
                "binary_bytes": rr.binary_bytes,
                "compile_ok": rr.compile_ok,
                "output": rr.output,
            }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON data written to: {path}")

# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="LCCC benchmark suite")
    p.add_argument("--reps",  type=int, default=5,    help="repetitions per benchmark (default 5)")
    p.add_argument("--bench", action="append",        help="run only this benchmark ID (repeatable)")
    p.add_argument("--md",    metavar="FILE",         help="write Markdown report to FILE")
    p.add_argument("--json",  metavar="FILE",         help="write JSON results to FILE")
    p.add_argument("--verbose", action="store_true",  help="show compile errors")
    args = p.parse_args()

    # Preflight checks
    missing = [name for name, (exe, _) in COMPILERS.items() if not Path(exe).exists()]
    if missing:
        print(c(RED, f"ERROR: missing compilers: {', '.join(missing)}"))
        sys.exit(1)
    if not BENCH_DIR.exists():
        print(c(RED, f"ERROR: benchmark dir not found: {BENCH_DIR}"))
        sys.exit(1)

    bench_ids = args.bench or []

    print(c(BOLD, "\nLCCC Benchmark Suite"))
    print(f"  reps per benchmark : {args.reps}")
    print(f"  LCCC               : {LCCC}")
    print(f"  CCC (upstream)     : {CCC}")
    print(f"  GCC                : {GCC}")

    results = run_all(bench_ids, args.reps, args.verbose)

    print_report(results)

    if args.md:
        write_markdown(results, args.md)
    if args.json:
        export_json(results, args.json)

if __name__ == "__main__":
    main()
