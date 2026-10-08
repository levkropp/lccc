#!/usr/bin/env python3
"""SQLite real-workload test: compile sqlite3.c and a workload harness with
LCCC, run, and diff the transcript against a GCC-built reference.

Covers multi-table transactions (with rollback), joins across three tables,
aggregates (GROUP BY/HAVING/min/max/avg/total), indexes, correlated
subqueries, and UPDATE/DELETE with WHERE.

Usage:
    python3 tests/integration/test_sqlite_workload.py

Requires the SQLite 3.45 amalgamation. Searched in order:
    $SQLITE_AMALG_DIR, /tmp/sqlite-amalgamation-3450000,
    tests/integration/sqlite-amalgamation-3450000
Fetch with:
    curl -sL https://www.sqlite.org/2024/sqlite-amalgamation-3450000.zip \\
        -o /tmp/sqlite.zip && python3 -c "import zipfile; \\
        zipfile.ZipFile('/tmp/sqlite.zip').extractall('/tmp')"

Environment:
    LCCC_BIN   path to the lccc binary (default: target/release/lccc)
    CCC_KEEP_SQLITE_OBJ=1   keep/reuse the compiled sqlite3 objects between
                            runs (the lccc compile takes minutes)
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LCCC = os.environ.get("LCCC_BIN", os.path.join(ROOT, "target", "release", "lccc"))
WORKLOAD = os.path.join(HERE, "sqlite_workload.c")
LINK_LIBS = ["-lm", "-ldl", "-lpthread"]

GCC_INC = subprocess.run(
    ["gcc", "-print-file-name=include"], capture_output=True, text=True
).stdout.strip()


def find_amalgamation():
    candidates = [
        os.environ.get("SQLITE_AMALG_DIR", ""),
        "/tmp/sqlite-amalgamation-3450000",
        os.path.join(HERE, "sqlite-amalgamation-3450000"),
    ]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "sqlite3.c")):
            return d
    return None


def compile_c(compiler, src, out_obj, inc_dir, timeout=900):
    # -fPIC: the Ubuntu default link is PIE, which rejects direct adrp+add
    # addressing of preemptible external symbols (e.g. dlsym in sqlite).
    cmd = [compiler, "-c", "-O2", "-fPIC", f"-I{inc_dir}", f"-I{GCC_INC}", src, "-o", out_obj]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{compiler} failed on {src}: {r.stderr[:400]}")


def main():
    amalg = find_amalgamation()
    if amalg is None:
        print("SKIP: sqlite3.c amalgamation not found (see docstring)")
        return 0
    sqlite_src = os.path.join(amalg, "sqlite3.c")

    tmp = tempfile.mkdtemp(prefix="sqlite_workload_")
    keep = os.environ.get("CCC_KEEP_SQLITE_OBJ") == "1"
    sqlite_gcc_obj = os.path.join(tmp, "sqlite3_gcc.o")
    sqlite_lccc_obj = os.path.join(tmp, "sqlite3_lccc.o")
    if keep:
        sqlite_gcc_obj = "/tmp/sqlite3_workload_gcc.o"
        sqlite_lccc_obj = "/tmp/sqlite3_workload_lccc.o"

    # Reference: sqlite and workload both built by GCC.
    if not (keep and os.path.isfile(sqlite_gcc_obj)):
        print("building reference sqlite3.o with GCC...", flush=True)
        compile_c("gcc", sqlite_src, sqlite_gcc_obj, amalg)
    print("building workload with GCC...", flush=True)
    wl_gcc_obj = os.path.join(tmp, "wl_gcc.o")
    compile_c("gcc", WORKLOAD, wl_gcc_obj, amalg)
    ref_bin = os.path.join(tmp, "ref")
    subprocess.run(
        ["gcc", wl_gcc_obj, sqlite_gcc_obj, "-o", ref_bin] + LINK_LIBS, check=True
    )
    ref = subprocess.run([ref_bin], capture_output=True, text=True, timeout=120)
    if ref.returncode != 0:
        print(f"FAIL: GCC reference run rc={ref.returncode}")
        return 1

    # Test: sqlite and workload both built by LCCC.
    if not (keep and os.path.isfile(sqlite_lccc_obj)):
        print("building sqlite3.o with LCCC (minutes)...", flush=True)
        compile_c(LCCC, sqlite_src, sqlite_lccc_obj, amalg)
    print("building workload with LCCC...", flush=True)
    wl_lccc_obj = os.path.join(tmp, "wl_lccc.o")
    compile_c(LCCC, WORKLOAD, wl_lccc_obj, amalg)
    test_bin = os.path.join(tmp, "test")
    subprocess.run(
        ["gcc", wl_lccc_obj, sqlite_lccc_obj, "-o", test_bin] + LINK_LIBS, check=True
    )
    got = subprocess.run([test_bin], capture_output=True, text=True, timeout=120)

    if got.returncode != 0:
        print(f"FAIL: LCCC build crashed rc={got.returncode}")
        print(got.stdout[:500])
        print(got.stderr[:500])
        return 1
    if got.stdout != ref.stdout:
        print("FAIL: output mismatch vs GCC reference")
        ref_lines = ref.stdout.splitlines()
        got_lines = got.stdout.splitlines()
        for i, (a, b) in enumerate(zip(ref_lines, got_lines)):
            if a != b:
                print(f"  first diff at line {i}: GCC={a!r} LCCC={b!r}")
                break
        print(f"  (ref {len(ref_lines)} lines, got {len(got_lines)} lines)")
        return 1
    if "FAIL" in got.stdout or "DONE" not in got.stdout:
        print("FAIL: workload reported an internal SQL error")
        print(got.stdout[:500])
        return 1

    print(f"PASS: sqlite workload transcript matches GCC ({len(ref.stdout.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
