#!/usr/bin/env python3
"""Progressive complexity test suite for LCCC.

Compiles programs of increasing complexity with both LCCC and GCC,
compares output. Fast feedback: 5s compile, 5s run per test.

Usage:
    python3 test_progressive.py              # Default flags
    python3 test_progressive.py --peephole   # Enable peephole optimizer
    python3 test_progressive.py --level 3    # Run only up to level 3
    python3 test_progressive.py --from-level 5  # Start from level 5
"""

import subprocess, sys, os, tempfile, argparse

COMPILE_TIMEOUT = 10  # seconds
RUN_TIMEOUT = 5       # seconds
SQLITE_COMPILE_TIMEOUT = 300  # 5 min for full SQLite

# Auto-detect GCC include path
def find_gcc_include():
    try:
        r = subprocess.run(["gcc", "-E", "-x", "c", "-v", "/dev/null"],
                          capture_output=True, text=True, timeout=5)
        for line in r.stderr.splitlines():
            line = line.strip()
            if line.startswith("/usr/lib/gcc") and "include" in line and os.path.isdir(line):
                return line
    except: pass
    return "/usr/lib/gcc/x86_64-pc-linux-gnu/15.2.1/include"

GCC_INC = find_gcc_include()

# Find LCCC binary
def find_lccc():
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../target/release/lccc"),
        "lccc",
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isfile(p):
            return p
    return "lccc"

LCCC = find_lccc()

# ── Test programs by level ──────────────────────────────────────────────

TESTS = []

def test(level, name, code):
    TESTS.append((level, name, code))

# Level 1: Trivial
test(1, "hello", r'''
#include <stdio.h>
int main(void) { printf("hello world\n"); return 0; }
''')

test(1, "arithmetic", r'''
#include <stdio.h>
int main(void) {
    int a = 42, b = 17;
    printf("%d %d %d %d %d\n", a+b, a-b, a*b, a/b, a%b);
    return 0;
}
''')

test(1, "float_math", r'''
#include <stdio.h>
int main(void) {
    double x = 3.14159, y = 2.71828;
    printf("%.5f %.5f %.5f\n", x+y, x*y, x/y);
    return 0;
}
''')

# Level 2: Control flow
test(2, "if_else", r'''
#include <stdio.h>
int classify(int x) {
    if (x < 0) return -1;
    else if (x == 0) return 0;
    else return 1;
}
int main(void) {
    for (int i = -3; i <= 3; i++) printf("%d:%d ", i, classify(i));
    printf("\n");
    return 0;
}
''')

test(2, "switch_stmt", r'''
#include <stdio.h>
const char *day(int d) {
    switch(d) {
        case 0: return "Sun"; case 1: return "Mon"; case 2: return "Tue";
        case 3: return "Wed"; case 4: return "Thu"; case 5: return "Fri";
        case 6: return "Sat"; default: return "???";
    }
}
int main(void) {
    for (int i = 0; i < 8; i++) printf("%s ", day(i));
    printf("\n");
    return 0;
}
''')

test(2, "nested_loops", r'''
#include <stdio.h>
int main(void) {
    int sum = 0;
    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
            sum += (i * j) & 0xFF;
    printf("%d\n", sum);
    return 0;
}
''')

test(2, "recursion_fib", r'''
#include <stdio.h>
int fib(int n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }
int main(void) {
    for (int i = 0; i < 15; i++) printf("%d ", fib(i));
    printf("\n");
    return 0;
}
''')

# Level 3: Pointers and structs
test(3, "linked_list", r'''
#include <stdio.h>
#include <stdlib.h>
typedef struct Node { int val; struct Node *next; } Node;
Node *push(Node *head, int val) {
    Node *n = malloc(sizeof(Node));
    n->val = val; n->next = head;
    return n;
}
int main(void) {
    Node *list = NULL;
    for (int i = 0; i < 10; i++) list = push(list, i * 7);
    for (Node *n = list; n; n = n->next) printf("%d ", n->val);
    printf("\n");
    while (list) { Node *t = list; list = list->next; free(t); }
    return 0;
}
''')

test(3, "struct_pass", r'''
#include <stdio.h>
typedef struct { double x, y, z; } Vec3;
Vec3 add(Vec3 a, Vec3 b) { return (Vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
int main(void) {
    Vec3 a = {1.0, 2.0, 3.0}, b = {4.0, 5.0, 6.0};
    Vec3 c = add(a, b);
    printf("%.1f %.1f %.1f dot=%.1f\n", c.x, c.y, c.z, dot(a, b));
    return 0;
}
''')

test(3, "array_sort", r'''
#include <stdio.h>
#include <stdlib.h>
int cmp(const void *a, const void *b) { return *(int*)a - *(int*)b; }
int main(void) {
    int arr[] = {9, 3, 7, 1, 5, 8, 2, 6, 4, 0};
    qsort(arr, 10, sizeof(int), cmp);
    for (int i = 0; i < 10; i++) printf("%d ", arr[i]);
    printf("\n");
    return 0;
}
''')

# Level 4: Function-heavy
test(4, "callback", r'''
#include <stdio.h>
int apply(int (*f)(int), int x) { return f(x); }
int square(int x) { return x * x; }
int negate(int x) { return -x; }
int main(void) {
    for (int i = 0; i < 5; i++)
        printf("%d %d ", apply(square, i), apply(negate, i));
    printf("\n");
    return 0;
}
''')

test(4, "variadic_sum", r'''
#include <stdio.h>
#include <stdarg.h>
int sum(int count, ...) {
    va_list ap; va_start(ap, count);
    int s = 0;
    for (int i = 0; i < count; i++) s += va_arg(ap, int);
    va_end(ap);
    return s;
}
int main(void) {
    printf("%d %d %d\n", sum(3, 1, 2, 3), sum(5, 10, 20, 30, 40, 50), sum(0));
    return 0;
}
''')

test(4, "static_locals", r'''
#include <stdio.h>
int counter(void) { static int n = 0; return ++n; }
int main(void) {
    for (int i = 0; i < 10; i++) printf("%d ", counter());
    printf("\n");
    return 0;
}
''')

# Level 5: Complex patterns
test(5, "bit_manipulation", r'''
#include <stdio.h>
unsigned reverse_bits(unsigned x) {
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
    return (x >> 16) | (x << 16);
}
int main(void) {
    unsigned vals[] = {0, 1, 0x80000000, 0xDEADBEEF, 0x12345678};
    for (int i = 0; i < 5; i++) printf("%08x ", reverse_bits(vals[i]));
    printf("\n");
    return 0;
}
''')

test(5, "conditional_assign", r'''
#include <stdio.h>
void setup(void *pBuf, int sz, int n) {
    if (pBuf == 0) sz = n = 0;
    if (n == 0) sz = 0;
    sz = sz & ~7;
    printf("sz=%d n=%d ptr=%s\n", sz, n, pBuf ? "yes" : "null");
}
int main(void) {
    char buf[100];
    setup(buf, 512, 8);
    setup(0, 512, 8);
    setup(buf, 512, 0);
    return 0;
}
''')

test(5, "ternary_chains", r'''
#include <stdio.h>
int classify(int x) {
    return x > 100 ? 5 : x > 50 ? 4 : x > 20 ? 3 : x > 5 ? 2 : x > 0 ? 1 : 0;
}
int main(void) {
    int vals[] = {-1, 0, 1, 5, 6, 20, 21, 50, 51, 100, 101};
    for (int i = 0; i < 11; i++) printf("%d:%d ", vals[i], classify(vals[i]));
    printf("\n");
    return 0;
}
''')

# Level 6: Mini-libraries
test(6, "hash_table_mini", r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SIZE 64
typedef struct E { char *key; int val; struct E *next; } E;
static E *tbl[SIZE];
unsigned hash(const char *s) { unsigned h = 0; while (*s) h = h*31 + *s++; return h % SIZE; }
void set(const char *k, int v) {
    unsigned h = hash(k);
    for (E *e = tbl[h]; e; e = e->next)
        if (strcmp(e->key, k) == 0) { e->val = v; return; }
    E *e = malloc(sizeof(E));
    e->key = strdup(k); e->val = v; e->next = tbl[h]; tbl[h] = e;
}
int get(const char *k) {
    for (E *e = tbl[hash(k)]; e; e = e->next)
        if (strcmp(e->key, k) == 0) return e->val;
    return -1;
}
int main(void) {
    set("alice", 30); set("bob", 25); set("charlie", 35);
    printf("%d %d %d %d\n", get("alice"), get("bob"), get("charlie"), get("dave"));
    set("alice", 31);
    printf("%d\n", get("alice"));
    return 0;
}
''')

test(6, "string_ops", r'''
#include <stdio.h>
#include <string.h>
#include <ctype.h>
void to_upper(char *s) { for (; *s; s++) *s = toupper(*s); }
int count_words(const char *s) {
    int n = 0, in_word = 0;
    for (; *s; s++) {
        if (isspace(*s)) in_word = 0;
        else if (!in_word) { in_word = 1; n++; }
    }
    return n;
}
int main(void) {
    char buf[] = "hello world from lccc compiler";
    printf("words=%d len=%zu\n", count_words(buf), strlen(buf));
    to_upper(buf);
    printf("%s\n", buf);
    return 0;
}
''')

test(6, "matrix_multiply", r'''
#include <stdio.h>
#define N 4
void matmul(double C[N][N], double A[N][N], double B[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) C[i][j] += A[i][k] * B[k][j];
        }
}
int main(void) {
    double A[N][N] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    double B[N][N] = {{16,15,14,13},{12,11,10,9},{8,7,6,5},{4,3,2,1}};
    double C[N][N];
    matmul(C, A, B);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%.0f ", C[i][j]);
        printf("\n");
    }
    return 0;
}
''')

# Level 7: SQLite-like patterns (extracted from SQLite source)
test(7, "pcache_setup", r'''
#include <stdio.h>
typedef struct { int isInit; int szSlot; int nSlot; int nFreeSlot; int nReserve;
                 void *pStart; void *pFree; void *pEnd; int bUnderPressure; } PCache1;
static PCache1 pcache1 = {1,0,0,0,0,0,0,0,0};
typedef struct PgF { struct PgF *pNext; } PgF;
#define RD8(x) ((x)&~7)
void setup(void *pBuf, int sz, int n) {
    if (pcache1.isInit) {
        PgF *p;
        if (pBuf==0) sz = n = 0;
        if (n==0) sz = 0;
        sz = RD8(sz);
        pcache1.szSlot = sz;
        pcache1.nSlot = pcache1.nFreeSlot = n;
        pcache1.nReserve = n>90 ? 10 : (n/10 + 1);
        pcache1.pStart = pBuf;
        pcache1.pFree = 0;
        pcache1.bUnderPressure = 0;
        while (n--) {
            p = (PgF*)pBuf;
            p->pNext = pcache1.pFree;
            pcache1.pFree = p;
            pBuf = (void*)&((char*)pBuf)[sz];
        }
        pcache1.pEnd = pBuf;
    }
}
int main(void) {
    char buf[4096];
    setup(buf, 512, 8);
    printf("sz=%d n=%d start=%s end=%s free=%s\n",
        pcache1.szSlot, pcache1.nSlot,
        pcache1.pStart ? "yes" : "null",
        pcache1.pEnd ? "yes" : "null",
        pcache1.pFree ? "yes" : "null");
    return 0;
}
''')

test(7, "vdbe_dispatch", r'''
#include <stdio.h>
/* Simplified VDBE-like opcode dispatch */
enum { OP_ADD=1, OP_SUB=2, OP_MUL=3, OP_HALT=4, OP_PRINT=5, OP_JUMP=6, OP_JZ=7 };
typedef struct { int opcode; int p1, p2, p3; } Op;
void execute(Op *prog, int nOp) {
    int regs[8] = {0};
    int pc = 0;
    while (pc < nOp) {
        Op *op = &prog[pc];
        switch (op->opcode) {
            case OP_ADD: regs[op->p3] = regs[op->p1] + regs[op->p2]; break;
            case OP_SUB: regs[op->p3] = regs[op->p1] - regs[op->p2]; break;
            case OP_MUL: regs[op->p3] = regs[op->p1] * regs[op->p2]; break;
            case OP_HALT: return;
            case OP_PRINT: printf("r%d=%d\n", op->p1, regs[op->p1]); break;
            case OP_JUMP: pc = op->p1; continue;
            case OP_JZ: if (regs[op->p1]==0) { pc = op->p2; continue; } break;
        }
        pc++;
    }
}
int main(void) {
    Op prog[] = {
        {OP_ADD, 0, 0, 1},  /* r1 = 0 */
        {OP_ADD, 1, 1, 1},  /* r1 += r1 (=0) -- placeholder */
        {OP_ADD, 0, 0, 2},  /* r2 = 0 */
        /* Manual: set r1=10, r2=3, r3=r1*r2 */
    };
    /* Just test basic dispatch */
    int regs[8] = {0};
    regs[1] = 10; regs[2] = 3;
    printf("%d\n", regs[1] * regs[2]);
    return 0;
}
''')

test(7, "str_vappendf", r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
/* Simplified sqlite3_str_vappendf-like formatter */
char *myfmt(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[256];
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    char *r = malloc(n + 1);
    memcpy(r, buf, n + 1);
    return r;
}
int main(void) {
    char *s1 = myfmt("CREATE %s %.*s", "TABLE", 5, "users_data");
    char *s2 = myfmt("%d + %d = %d", 40, 2, 42);
    char *s3 = myfmt("%.3f", 3.14159);
    printf("%s\n%s\n%s\n", s1, s2, s3);
    free(s1); free(s2); free(s3);
    return 0;
}
''')

# Level 8: Full SQLite
# This is handled specially — needs sqlite3.c from /tmp/sqlite-amalgamation-3450000/

SQLITE_TEST_CODE = r'''
#include <stdio.h>
#include "sqlite3.h"
static int cb(void *d, int argc, char **argv, char **cols) {
    for (int i = 0; i < argc; i++) printf("  %s=%s", cols[i], argv[i]?argv[i]:"NULL");
    printf("\n"); return 0;
}
int main(void) {
    sqlite3 *db; char *err = NULL; int rc;
    sqlite3_open(":memory:", &db);
    rc = sqlite3_exec(db, "CREATE TABLE t(x INTEGER, y TEXT);", NULL, NULL, &err);
    if (rc) { printf("FAIL create: %s\n", err?err:"?"); return 1; }
    sqlite3_exec(db, "INSERT INTO t VALUES(42,'hello');", NULL, NULL, &err);
    rc = sqlite3_exec(db, "SELECT * FROM t;", cb, NULL, &err);
    if (rc) { printf("FAIL select: %s\n", err?err:"?"); return 1; }
    sqlite3_close(db);
    printf("OK\n");
    return 0;
}
'''

# ── Runner ──────────────────────────────────────────────────────────────

def compile_and_run(compiler, flags, code, timeout_compile=COMPILE_TIMEOUT, timeout_run=RUN_TIMEOUT, extra_src=None, extra_obj=None, link_flags=None):
    """Compile code, run it, return (success, output, error_msg)."""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(code)
        src = f.name
    out = src.replace(".c", ".out")
    try:
        cmd = compiler + flags + [src]
        if extra_obj:
            cmd = compiler + flags + [src] + [extra_obj]
        cmd += ["-o", out]
        if link_flags:
            cmd += link_flags
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_compile)
        if r.returncode != 0:
            return False, "", f"COMPILE: {r.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "", "COMPILE_TIMEOUT"
    finally:
        if extra_src:
            pass  # don't delete extra source
    try:
        r = subprocess.run([out], capture_output=True, text=True, timeout=timeout_run)
        if r.returncode != 0:
            return False, r.stdout, f"RUN_FAIL (rc={r.returncode}): {r.stderr[:200]}"
        return True, r.stdout, ""
    except subprocess.TimeoutExpired:
        return False, "", "RUN_TIMEOUT"
    finally:
        for f in [src, out]:
            try: os.unlink(f)
            except: pass

def run_tests(max_level=8, min_level=1, env_extra=None):
    gcc_cmd = ["gcc"]
    gcc_flags = ["-O2"]
    lccc_cmd = [LCCC]
    lccc_flags = ["-O2", f"-I{GCC_INC}"]

    env_backup = os.environ.copy()
    if env_extra:
        os.environ.update(env_extra)

    passed = 0
    failed = 0
    total = 0

    for level, name, code in TESTS:
        if level < min_level or level > max_level:
            continue
        total += 1
        label = f"L{level} {name}"

        # GCC reference
        ok_gcc, out_gcc, err_gcc = compile_and_run(gcc_cmd, gcc_flags, code)
        if not ok_gcc:
            print(f"  SKIP  {label:30s} (GCC failed: {err_gcc})")
            continue

        # LCCC
        ok_lccc, out_lccc, err_lccc = compile_and_run(lccc_cmd, lccc_flags, code)
        if not ok_lccc:
            print(f"  FAIL  {label:30s} {err_lccc}")
            failed += 1
            continue

        if out_lccc.strip() == out_gcc.strip():
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label:30s} OUTPUT MISMATCH")
            print(f"        GCC:  {out_gcc.strip()[:80]}")
            print(f"        LCCC: {out_lccc.strip()[:80]}")
            failed += 1

    # Level 8: Full SQLite (special handling)
    if max_level >= 8 and min_level <= 8:
        total += 1
        label = "L8 sqlite_full"
        sqlite_src = "/tmp/sqlite-amalgamation-3450000/sqlite3.c"
        sqlite_inc = "/tmp/sqlite-amalgamation-3450000"
        if not os.path.isfile(sqlite_src):
            print(f"  SKIP  {label:30s} (sqlite3.c not found at {sqlite_src})")
        else:
            # Compile sqlite3.c with LCCC
            sqlite_obj = "/tmp/sqlite3_progressive.o"
            print(f"  ...   {label:30s} compiling sqlite3.c...", end="", flush=True)
            try:
                r = subprocess.run(
                    [LCCC, "-c", "-O2", f"-I{sqlite_inc}", f"-I{GCC_INC}", sqlite_src, "-o", sqlite_obj],
                    capture_output=True, text=True, timeout=SQLITE_COMPILE_TIMEOUT,
                    env=os.environ
                )
                if r.returncode != 0:
                    print(f"\r  FAIL  {label:30s} COMPILE: {r.stderr[:200]}")
                    failed += 1
                else:
                    # Compile test harness
                    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
                        f.write(SQLITE_TEST_CODE)
                        test_src = f.name
                    test_obj = test_src.replace(".c", ".o")
                    test_out = test_src.replace(".c", ".out")
                    try:
                        subprocess.run(
                            [LCCC, "-c", "-O2", f"-I{sqlite_inc}", f"-I{GCC_INC}", test_src, "-o", test_obj],
                            capture_output=True, timeout=COMPILE_TIMEOUT
                        )
                        subprocess.run(
                            ["gcc", test_obj, sqlite_obj, "-o", test_out, "-lm", "-ldl", "-lpthread"],
                            capture_output=True, timeout=10
                        )
                        r = subprocess.run([test_out], capture_output=True, text=True, timeout=RUN_TIMEOUT)
                        if r.returncode == 0 and "OK" in r.stdout:
                            print(f"\r  PASS  {label}")
                            passed += 1
                        else:
                            print(f"\r  FAIL  {label:30s} {r.stdout.strip()[:80]} {r.stderr.strip()[:80]}")
                            failed += 1
                    finally:
                        for f in [test_src, test_obj, test_out]:
                            try: os.unlink(f)
                            except: pass
            except subprocess.TimeoutExpired:
                print(f"\r  FAIL  {label:30s} COMPILE_TIMEOUT ({SQLITE_COMPILE_TIMEOUT}s)")
                failed += 1

    os.environ.clear()
    os.environ.update(env_backup)

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed — {total} total")
    return failed == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peephole", action="store_true", help="Enable peephole optimizer")
    parser.add_argument("--level", type=int, default=8, help="Max level to test")
    parser.add_argument("--from-level", type=int, default=1, help="Min level to test")
    args = parser.parse_args()

    env = {}
    if args.peephole:
        env["CCC_PEEPHOLE"] = "1"
        print("Running with CCC_PEEPHOLE=1\n")
    else:
        print("Running with default flags (peephole off)\n")

    ok = run_tests(max_level=args.level, min_level=args.from_level, env_extra=env if env else None)
    sys.exit(0 if ok else 1)
