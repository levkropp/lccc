#!/usr/bin/env python3
"""
LCCC Correctness Test Suite
Compiles C programs with LCCC and GCC, verifies identical output.
Tests cover edge cases, integer overflow, bitops, unions, enums,
multi-file, preprocessor, and other tricky patterns.

Usage:
    python3 run_correctness.py           # run all
    python3 run_correctness.py -v        # verbose (show failures)
    python3 run_correctness.py --filter bitfield  # filter by name
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
LCCC = REPO_ROOT / "target" / "release" / "lccc"
GCC_INC = subprocess.check_output(
    ["gcc", "-print-file-name=include"], text=True
).strip()

COMPILERS = {
    "LCCC": (str(LCCC), [f"-I{GCC_INC}", "-O2"]),
    "GCC": ("/usr/bin/gcc", ["-O2"]),
}

BOLD = "\033[1m"; DIM = "\033[2m"; RED = "\033[31m"
GRN = "\033[32m"; YLW = "\033[33m"; RST = "\033[0m"

def c(code, text):
    return f"{code}{text}{RST}" if sys.stdout.isatty() else text

# Each test: (name, source_code, extra_compile_flags, expected_exit_code)
# If expected_exit_code is None, we just check LCCC output == GCC output
TESTS = [
    # ── Integer edge cases ──────────────────────────────────────────────
    ("int_overflow_signed", r'''
#include <stdio.h>
#include <limits.h>
int main(void) {
    unsigned int a = UINT_MAX;
    printf("UINT_MAX=%u UINT_MAX+1=%u\n", a, a + 1);
    int b = INT_MAX;
    unsigned int c = (unsigned int)b + 1;
    printf("INT_MAX=%d cast=%u\n", b, c);
    printf("INT_MIN=%d\n", INT_MIN);
    return 0;
}
''', [], None),

    ("unsigned_division", r'''
#include <stdio.h>
int main(void) {
    unsigned int a = 0xFFFFFFFF;
    unsigned int b = 3;
    printf("%u / %u = %u rem %u\n", a, b, a/b, a%b);
    unsigned long long c = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long d = 7;
    printf("%llu / %llu = %llu rem %llu\n", c, d, c/d, c%d);
    return 0;
}
''', [], None),

    ("signed_division_negative", r'''
#include <stdio.h>
int main(void) {
    printf("%d %d\n", -7 / 2, -7 % 2);
    printf("%d %d\n", 7 / -2, 7 % -2);
    printf("%d %d\n", -7 / -2, -7 % -2);
    long long a = -9223372036854775807LL - 1;
    printf("%lld / -1 is tricky\n", a);
    return 0;
}
''', [], None),

    ("shift_edge_cases", r'''
#include <stdio.h>
int main(void) {
    unsigned int a = 1;
    printf("%u %u %u\n", a << 0, a << 15, a << 31);
    unsigned int b = 0x80000000;
    printf("%u %u\n", b >> 0, b >> 31);
    int c = -1;
    printf("%d %d\n", c >> 1, c >> 31);  // arithmetic shift
    return 0;
}
''', [], None),

    ("integer_conversions", r'''
#include <stdio.h>
int main(void) {
    // Narrowing
    int a = 0x12345678;
    short b = (short)a;
    char d = (char)a;
    printf("narrow: %hd %hhd\n", b, d);
    // Widening
    signed char e = -42;
    int f = e;
    unsigned int g = (unsigned int)e;
    printf("widen: %d %u\n", f, g);
    // Zero-extend vs sign-extend
    unsigned char h = 200;
    int i = h;
    int j = (signed char)h;
    printf("extend: %d %d\n", i, j);
    return 0;
}
''', [], None),

    # ── Bitfield tests ──────────────────────────────────────────────────
    ("bitfield_basic", r'''
#include <stdio.h>
struct S {
    unsigned int a : 3;
    unsigned int b : 5;
    unsigned int c : 8;
    unsigned int d : 16;
};
int main(void) {
    struct S s = {7, 31, 255, 65535};
    printf("%u %u %u %u\n", s.a, s.b, s.c, s.d);
    s.a = 0; s.b = 1;
    printf("%u %u\n", s.a, s.b);
    printf("sizeof=%zu\n", sizeof(struct S));
    return 0;
}
''', [], None),

    ("bitfield_signed", r'''
#include <stdio.h>
struct S {
    int a : 4;
    int b : 1;
    unsigned int c : 3;
};
int main(void) {
    struct S s;
    s.a = -4; s.b = -1; s.c = 7;
    printf("%d %d %u\n", s.a, s.b, s.c);
    s.a = 7;
    printf("%d\n", s.a);  // 7 in 4-bit signed = 7
    return 0;
}
''', [], None),

    # ── Union tests ─────────────────────────────────────────────────────
    ("union_type_punning", r'''
#include <stdio.h>
#include <string.h>
union U { int i; float f; unsigned char bytes[4]; };
int main(void) {
    union U u;
    u.i = 0x40490FDB;  // pi as float bits
    printf("int=%d float=%.4f\n", u.i, u.f);
    u.f = 1.0f;
    printf("bytes:");
    for (int k = 0; k < 4; k++) printf(" %02x", u.bytes[k]);
    printf("\n");
    return 0;
}
''', [], None),

    ("union_sizeof", r'''
#include <stdio.h>
union A { char c; int i; double d; };
union B { char arr[20]; int i; };
int main(void) {
    printf("A=%zu B=%zu\n", sizeof(union A), sizeof(union B));
    return 0;
}
''', [], None),

    # ── Enum tests ──────────────────────────────────────────────────────
    ("enum_values", r'''
#include <stdio.h>
enum Color { RED, GREEN = 5, BLUE, YELLOW = -1 };
enum Big { A = 100000000, B };
int main(void) {
    printf("%d %d %d %d\n", RED, GREEN, BLUE, YELLOW);
    printf("%d %d\n", A, B);
    enum Color c = BLUE;
    printf("blue=%d size=%zu\n", c, sizeof(c));
    return 0;
}
''', [], None),

    # ── Typedef and complex types ───────────────────────────────────────
    ("typedef_complex", r'''
#include <stdio.h>
typedef int (*BinOp)(int, int);
typedef struct { BinOp ops[4]; int count; } OpTable;

static int add(int a, int b) { return a + b; }
static int sub(int a, int b) { return a - b; }
static int mul(int a, int b) { return a * b; }
static int my_div(int a, int b) { return a / b; }

int main(void) {
    OpTable t = {{add, sub, mul, my_div}, 4};
    for (int i = 0; i < t.count; i++)
        printf("%d ", t.ops[i](10, 3));
    printf("\n");
    return 0;
}
''', [], None),

    # ── Array and pointer patterns ──────────────────────────────────────
    ("multidim_array", r'''
#include <stdio.h>
int main(void) {
    int a[3][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    int sum = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            sum += a[i][j];
    printf("sum=%d a[1][2]=%d\n", sum, a[1][2]);
    int (*p)[4] = a;
    printf("p[2][3]=%d\n", p[2][3]);
    return 0;
}
''', [], None),

    ("array_decay", r'''
#include <stdio.h>
void print_arr(int *p, int n) {
    for (int i = 0; i < n; i++) printf("%d ", p[i]);
    printf("\n");
}
int main(void) {
    int a[] = {10, 20, 30, 40, 50};
    print_arr(a, 5);
    print_arr(a + 2, 3);
    printf("diff=%td\n", (a+4) - (a+1));
    return 0;
}
''', [], None),

    ("void_pointer_arith", r'''
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    int arr[5] = {1, 2, 3, 4, 5};
    void *p = arr;
    // Cast through char* for arithmetic
    int *q = (int *)((char *)p + 2 * sizeof(int));
    printf("%d\n", *q);
    return 0;
}
''', [], None),

    # ── String literal and const ────────────────────────────────────────
    ("string_literals", r'''
#include <stdio.h>
#include <string.h>
int main(void) {
    const char *s1 = "hello";
    const char *s2 = "hello";
    printf("eq=%d len=%zu\n", s1 == s2, strlen(s1));
    char buf[] = "world";
    buf[0] = 'W';
    printf("%s\n", buf);
    // Concatenation
    printf("%s\n", "foo" "bar" "baz");
    // Escape sequences
    printf("%d %d %d\n", '\n', '\t', '\0');
    return 0;
}
''', [], None),

    # ── Control flow edge cases ─────────────────────────────────────────
    ("goto_label", r'''
#include <stdio.h>
int main(void) {
    int i = 0;
    loop:
    if (i >= 5) goto done;
    printf("%d ", i);
    i++;
    goto loop;
    done:
    printf("\n");
    return 0;
}
''', [], None),

    ("nested_switch", r'''
#include <stdio.h>
int main(void) {
    for (int i = 0; i < 3; i++) {
        switch (i) {
            case 0:
                switch (i + 1) {
                    case 1: printf("0-1 "); break;
                    default: printf("0-? "); break;
                }
                break;
            case 1: printf("1 "); break;
            case 2: printf("2 "); break;
        }
    }
    printf("\n");
    return 0;
}
''', [], None),

    ("switch_fallthrough", r'''
#include <stdio.h>
int main(void) {
    for (int i = 0; i < 5; i++) {
        int v = 0;
        switch (i) {
            case 0: v += 1;
            case 1: v += 10;
            case 2: v += 100; break;
            case 3: v += 1000; break;
            default: v = -1;
        }
        printf("%d ", v);
    }
    printf("\n");
    return 0;
}
''', [], None),

    ("do_while", r'''
#include <stdio.h>
int main(void) {
    int i = 0, sum = 0;
    do { sum += i; i++; } while (i < 10);
    printf("sum=%d\n", sum);
    // do-while with break
    i = 0;
    do { if (i == 5) break; i++; } while (1);
    printf("i=%d\n", i);
    return 0;
}
''', [], None),

    ("comma_operator", r'''
#include <stdio.h>
int main(void) {
    int a = (1, 2, 3);
    printf("%d\n", a);
    int b = 0, d = 0;
    for (int i = 0, j = 10; i < 5; i++, j--) {
        b += i; d += j;
    }
    printf("%d %d\n", b, d);
    return 0;
}
''', [], None),

    ("ternary_nested", r'''
#include <stdio.h>
int main(void) {
    for (int i = 0; i < 10; i++) {
        int v = i < 3 ? 1 : i < 6 ? 2 : i < 8 ? 3 : 4;
        printf("%d ", v);
    }
    printf("\n");
    return 0;
}
''', [], None),

    # ── Struct/memory patterns ──────────────────────────────────────────
    ("struct_padding", r'''
#include <stdio.h>
struct A { char c; int i; char d; };
struct B { int i; char c; char d; };
struct C { char c; short s; int i; };
int main(void) {
    printf("A=%zu B=%zu C=%zu\n", sizeof(struct A), sizeof(struct B), sizeof(struct C));
    return 0;
}
''', [], None),

    ("struct_return_large", r'''
#include <stdio.h>
typedef struct { long a, b, c, d; } Big;
Big make_big(int x) {
    Big b = { x, x*2, x*3, x*4 };
    return b;
}
int main(void) {
    Big b = make_big(10);
    printf("%ld %ld %ld %ld\n", b.a, b.b, b.c, b.d);
    return 0;
}
''', [], None),

    ("flexible_array_member", r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct Msg { int len; char data[]; };
int main(void) {
    struct Msg *m = malloc(sizeof(struct Msg) + 6);
    m->len = 5;
    memcpy(m->data, "hello", 6);
    printf("len=%d data=%s\n", m->len, m->data);
    free(m);
    return 0;
}
''', [], None),

    ("nested_struct_init", r'''
#include <stdio.h>
struct Inner { int x, y; };
struct Outer { struct Inner a; struct Inner b; int z; };
int main(void) {
    struct Outer o = {{1, 2}, {3, 4}, 5};
    printf("%d %d %d %d %d\n", o.a.x, o.a.y, o.b.x, o.b.y, o.z);
    return 0;
}
''', [], None),

    # ── Floating point edge cases ───────────────────────────────────────
    ("float_special", r'''
#include <stdio.h>
#include <math.h>
int main(void) {
    double inf = 1.0 / 0.0;
    double nan_val = 0.0 / 0.0;
    printf("inf=%f -inf=%f\n", inf, -inf);
    printf("nan=%d inf=%d\n", isnan(nan_val), isinf(inf));
    printf("1.0/3.0=%.17g\n", 1.0/3.0);
    // Float vs double precision
    float f = 1.0f / 3.0f;
    double d = 1.0 / 3.0;
    printf("f=%.9g d=%.17g\n", f, d);
    return 0;
}
''', ["-lm"], None),

    ("float_cast", r'''
#include <stdio.h>
int main(void) {
    double d = 3.99;
    int i = (int)d;
    printf("d=%.2f i=%d\n", d, i);
    d = -3.99;
    i = (int)d;
    printf("d=%.2f i=%d\n", d, i);
    float f = 1e10f;
    long long ll = (long long)f;
    printf("f=%g ll=%lld\n", f, ll);
    unsigned int u = 3000000000u;
    double d2 = (double)u;
    printf("u=%u d=%.0f\n", u, d2);
    return 0;
}
''', [], None),

    # ── Preprocessor patterns ───────────────────────────────────────────
    ("preprocessor_stringize", r'''
#include <stdio.h>
#define STR(x) #x
#define CONCAT(a, b) a##b
#define XSTR(x) STR(x)
#define VERSION 42
int main(void) {
    printf("%s\n", STR(hello world));
    int CONCAT(my, var) = 10;
    printf("%d\n", myvar);
    printf("version=%s\n", XSTR(VERSION));
    return 0;
}
''', [], None),

    ("preprocessor_variadic", r'''
#include <stdio.h>
#define LOG(fmt, ...) printf("[LOG] " fmt "\n", ##__VA_ARGS__)
int main(void) {
    LOG("hello %s", "world");
    LOG("number %d and %d", 1, 2);
    LOG("no args");
    return 0;
}
''', [], None),

    # ── Recursion and stack patterns ────────────────────────────────────
    ("mutual_recursion", r'''
#include <stdio.h>
static int is_even(int n);
static int is_odd(int n);
static int is_even(int n) { return n == 0 ? 1 : is_odd(n - 1); }
static int is_odd(int n)  { return n == 0 ? 0 : is_even(n - 1); }
int main(void) {
    for (int i = 0; i < 10; i++)
        printf("%d:%d ", i, is_even(i));
    printf("\n");
    return 0;
}
''', [], None),

    # ── VLA (variable-length array) ─────────────────────────────────────
    ("vla_basic", r'''
#include <stdio.h>
void fill(int n) {
    int arr[n];
    for (int i = 0; i < n; i++) arr[i] = i * i;
    int sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    printf("n=%d sum=%d\n", n, sum);
}
int main(void) {
    fill(5);
    fill(10);
    fill(1);
    return 0;
}
''', [], None),

    # ── Static and global patterns ──────────────────────────────────────
    ("static_local", r'''
#include <stdio.h>
int counter(void) {
    static int n = 0;
    return ++n;
}
int main(void) {
    for (int i = 0; i < 5; i++)
        printf("%d ", counter());
    printf("\n");
    return 0;
}
''', [], None),

    ("global_init_order", r'''
#include <stdio.h>
int a = 10;
int b = 20;
int arr[] = {1, 2, 3, 4, 5};
const char *msg = "hello";
int main(void) {
    printf("a=%d b=%d arr[2]=%d msg=%s\n", a, b, arr[2], msg);
    a += b;
    printf("a=%d\n", a);
    return 0;
}
''', [], None),

    # ── Alignment and sizeof ────────────────────────────────────────────
    ("sizeof_types", r'''
#include <stdio.h>
int main(void) {
    printf("char=%zu short=%zu int=%zu long=%zu llong=%zu\n",
        sizeof(char), sizeof(short), sizeof(int), sizeof(long), sizeof(long long));
    printf("float=%zu double=%zu ldouble=%zu\n",
        sizeof(float), sizeof(double), sizeof(long double));
    printf("ptr=%zu\n", sizeof(void *));
    return 0;
}
''', [], None),

    # ── Compound literal ────────────────────────────────────────────────
    ("compound_literal", r'''
#include <stdio.h>
struct Point { int x, y; };
void print_point(struct Point p) { printf("(%d,%d)\n", p.x, p.y); }
int main(void) {
    print_point((struct Point){3, 4});
    int *p = (int[]){10, 20, 30};
    printf("%d %d %d\n", p[0], p[1], p[2]);
    return 0;
}
''', [], None),

    # ── Designated initializer ──────────────────────────────────────────
    ("designated_init", r'''
#include <stdio.h>
struct S { int a, b, c, d; };
int main(void) {
    struct S s = { .c = 30, .a = 10 };
    printf("%d %d %d %d\n", s.a, s.b, s.c, s.d);
    int arr[10] = { [3] = 33, [7] = 77 };
    for (int i = 0; i < 10; i++) printf("%d ", arr[i]);
    printf("\n");
    return 0;
}
''', [], None),

    # ── Long long arithmetic ────────────────────────────────────────────
    ("long_long_arith", r'''
#include <stdio.h>
int main(void) {
    long long a = 1000000000LL * 1000000000LL;
    printf("%lld\n", a);
    unsigned long long b = 18446744073709551615ULL;
    printf("%llu\n", b);
    printf("%llu\n", b / 3);
    long long c = -9223372036854775807LL - 1;
    printf("%lld\n", c);
    return 0;
}
''', [], None),

    # ── Inline functions ────────────────────────────────────────────────
    ("inline_func", r'''
#include <stdio.h>
static inline int square(int x) { return x * x; }
static inline int max(int a, int b) { return a > b ? a : b; }
int main(void) {
    int sum = 0;
    for (int i = 0; i < 10; i++)
        sum += square(i) + max(i, 5);
    printf("sum=%d\n", sum);
    return 0;
}
''', [], None),

    # ── Volatile ────────────────────────────────────────────────────────
    ("volatile_access", r'''
#include <stdio.h>
int main(void) {
    volatile int x = 42;
    int y = x;
    x = y + 1;
    printf("%d\n", (int)x);
    volatile int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) sum += arr[i];
    printf("%d\n", sum);
    return 0;
}
''', [], None),

    # ── Complex expressions ─────────────────────────────────────────────
    ("complex_expressions", r'''
#include <stdio.h>
int main(void) {
    int a = 5, b = 3;
    // Chained comparisons via &&
    printf("%d\n", (a > b) && (b > 1) && (a < 10));
    // Short-circuit
    int x = 0;
    int r = (x != 0) && (10 / x > 0);
    printf("%d\n", r);
    r = (x == 0) || (10 / x > 0);
    printf("%d\n", r);
    // Conditional with side effects
    int c = 0;
    (a > b) ? (c = 1) : (c = 2);
    printf("%d\n", c);
    return 0;
}
''', [], None),

    # ── Pointer to function returning pointer ───────────────────────────
    ("complex_decl", r'''
#include <stdio.h>
static int val = 42;
int *get_ptr(void) { return &val; }
int main(void) {
    int *(*fp)(void) = get_ptr;
    printf("%d\n", *fp());
    // Array of function pointers
    int *(*fps[1])(void) = { get_ptr };
    printf("%d\n", *fps[0]());
    return 0;
}
''', [], None),

    # ── Multi-file compilation ──────────────────────────────────────────
    # This one is special - we handle it separately in the runner

    # ── Large switch (jump table test) ──────────────────────────────────
    ("large_switch", r'''
#include <stdio.h>
int decode(int x) {
    switch (x) {
        case 0: return 100;  case 1: return 101;  case 2: return 104;
        case 3: return 109;  case 4: return 116;  case 5: return 125;
        case 6: return 136;  case 7: return 149;  case 8: return 164;
        case 9: return 181;  case 10: return 200; case 11: return 221;
        case 12: return 244; case 13: return 269; case 14: return 296;
        case 15: return 325; case 16: return 356; case 17: return 389;
        case 18: return 424; case 19: return 461; default: return -1;
    }
}
int main(void) {
    int sum = 0;
    for (int i = -1; i <= 20; i++) sum += decode(i);
    printf("sum=%d\n", sum);
    return 0;
}
''', [], None),

    # ── Recursive struct (linked list variants) ─────────────────────────
    ("doubly_linked_list", r'''
#include <stdio.h>
#include <stdlib.h>
typedef struct Node { int val; struct Node *prev, *next; } Node;
int main(void) {
    Node *head = NULL, *tail = NULL;
    for (int i = 1; i <= 5; i++) {
        Node *n = (Node *)malloc(sizeof(Node));
        n->val = i; n->next = NULL; n->prev = tail;
        if (tail) tail->next = n; else head = n;
        tail = n;
    }
    // Forward
    for (Node *p = head; p; p = p->next) printf("%d ", p->val);
    printf("| ");
    // Backward
    for (Node *p = tail; p; p = p->prev) printf("%d ", p->val);
    printf("\n");
    // Cleanup
    Node *p = head;
    while (p) { Node *next = p->next; free(p); p = next; }
    return 0;
}
''', [], None),

    # ── Variadic function edge cases ────────────────────────────────────
    ("varargs_mixed_types", r'''
#include <stdio.h>
#include <stdarg.h>
double sum_mixed(int count, ...) {
    va_list ap;
    va_start(ap, count);
    double total = 0;
    for (int i = 0; i < count; i++) {
        if (i % 2 == 0)
            total += va_arg(ap, int);
        else
            total += va_arg(ap, double);
    }
    va_end(ap);
    return total;
}
int main(void) {
    printf("%.1f\n", sum_mixed(4, 10, 2.5, 20, 3.5));
    printf("%.1f\n", sum_mixed(2, 100, 0.5));
    return 0;
}
''', [], None),

    # ── Snprintf / sprintf ──────────────────────────────────────────────
    ("snprintf_test", r'''
#include <stdio.h>
int main(void) {
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%d + %d = %d", 10, 20, 30);
    printf("%s (len=%d)\n", buf, n);
    n = snprintf(buf, 10, "long string that gets truncated");
    printf("%s (len=%d)\n", buf, n);
    return 0;
}
''', [], None),

    # ── Sizeof expression vs type ───────────────────────────────────────
    ("sizeof_expr", r'''
#include <stdio.h>
int main(void) {
    int arr[10];
    printf("arr=%zu elem=%zu count=%zu\n",
           sizeof(arr), sizeof(arr[0]), sizeof(arr) / sizeof(arr[0]));
    char *p = "hello";
    printf("ptr=%zu\n", sizeof(p));
    printf("literal=%zu\n", sizeof("hello"));
    // sizeof doesn't evaluate
    int x = 5;
    printf("before=%d sizeof=%zu after=%d\n", x, sizeof(x++), x);
    return 0;
}
''', [], None),

    # ── Null pointer patterns ───────────────────────────────────────────
    ("null_pointer", r'''
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    int *p = NULL;
    printf("null=%d\n", p == NULL);
    printf("null=%d\n", p == 0);
    printf("notnull=%d\n", p != NULL);
    void *v = NULL;
    printf("void_null=%d\n", v == NULL);
    // Null function pointer
    int (*fp)(void) = NULL;
    printf("fp_null=%d\n", fp == NULL);
    return 0;
}
''', [], None),

    # ── Cast between pointer types ──────────────────────────────────────
    ("pointer_casts", r'''
#include <stdio.h>
#include <stdint.h>
int main(void) {
    int x = 0x41424344;
    char *cp = (char *)&x;
    printf("first byte: %d\n", *cp);
    // Round-trip pointer -> int -> pointer
    intptr_t ip = (intptr_t)&x;
    int *p2 = (int *)ip;
    printf("round trip: %d\n", *p2 == x);
    return 0;
}
''', [], None),

    # ── Array of strings ────────────────────────────────────────────────
    ("string_array", r'''
#include <stdio.h>
int main(void) {
    const char *days[] = {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
    for (int i = 0; i < 7; i++) printf("%s ", days[i]);
    printf("\n");
    // 2D char array
    char names[][10] = {"Alice", "Bob", "Charlie"};
    for (int i = 0; i < 3; i++) printf("%s ", names[i]);
    printf("\n");
    return 0;
}
''', [], None),
]

# Multi-file test (handled specially)
MULTI_FILE_A = r'''
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }
'''

MULTI_FILE_B = r'''
#include <stdio.h>
extern int add(int, int);
extern int mul(int, int);
int main(void) {
    printf("%d %d\n", add(3, 4), mul(3, 4));
    return 0;
}
'''

def run_test(name, src, extra_flags, verbose, tmpdir):
    """Compile and run a single test with both compilers, compare output."""
    tmp = Path(tmpdir)
    src_file = tmp / f"{name}.c"
    src_file.write_text(src)

    results = {}
    for cname, (exe, flags) in COMPILERS.items():
        out = tmp / f"{name}_{cname}"
        cmd = [exe] + flags + extra_flags + ["-o", str(out), str(src_file)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            results[cname] = ("COMPILE_TIMEOUT", "", "")
            continue
        if r.returncode != 0:
            results[cname] = ("COMPILE_FAIL", r.stderr.strip(), "")
            continue
        try:
            r2 = subprocess.run([str(out)], capture_output=True, text=True, timeout=10)
            results[cname] = ("OK", r2.stdout.strip(), r2.stderr.strip())
        except subprocess.TimeoutExpired:
            results[cname] = ("RUN_TIMEOUT", "", "")
        except Exception as e:
            results[cname] = ("RUN_ERROR", str(e), "")

    gcc_status, gcc_out, _ = results.get("GCC", ("MISSING", "", ""))
    lccc_status, lccc_out, lccc_err = results.get("LCCC", ("MISSING", "", ""))

    if gcc_status != "OK":
        return "SKIP", f"GCC {gcc_status}"
    if lccc_status == "COMPILE_FAIL":
        return "COMPILE_FAIL", results["LCCC"][1][:200]
    if lccc_status == "COMPILE_TIMEOUT":
        return "COMPILE_TIMEOUT", ""
    if lccc_status == "RUN_TIMEOUT":
        return "RUN_TIMEOUT", ""
    if lccc_status == "RUN_ERROR":
        return "RUN_ERROR", results["LCCC"][1]
    if lccc_status != "OK":
        return "ERROR", lccc_status
    if gcc_out == lccc_out:
        return "PASS", ""
    return "MISMATCH", f"GCC: {gcc_out[:80]}\nLCCC: {lccc_out[:80]}"

def run_multi_file_test(verbose, tmpdir):
    """Test multi-file compilation (separate compile + link)."""
    tmp = Path(tmpdir)
    a_src = tmp / "multi_a.c"
    b_src = tmp / "multi_b.c"
    a_src.write_text(MULTI_FILE_A)
    b_src.write_text(MULTI_FILE_B)

    for cname, (exe, flags) in COMPILERS.items():
        a_obj = tmp / f"multi_a_{cname}.o"
        b_obj = tmp / f"multi_b_{cname}.o"
        out = tmp / f"multi_{cname}"

        # Compile each file separately
        r1 = subprocess.run([exe] + flags + ["-c", "-o", str(a_obj), str(a_src)],
                           capture_output=True, text=True, timeout=30)
        if r1.returncode != 0:
            if cname == "LCCC":
                return "COMPILE_FAIL", f"LCCC -c failed: {r1.stderr[:100]}"
            return "SKIP", "GCC -c failed"

        r2 = subprocess.run([exe] + flags + ["-c", "-o", str(b_obj), str(b_src)],
                           capture_output=True, text=True, timeout=30)
        if r2.returncode != 0:
            if cname == "LCCC":
                return "COMPILE_FAIL", f"LCCC -c failed: {r2.stderr[:100]}"
            return "SKIP", "GCC -c failed"

        # Link
        r3 = subprocess.run([exe] + flags + ["-o", str(out), str(a_obj), str(b_obj)],
                           capture_output=True, text=True, timeout=30)
        if r3.returncode != 0:
            if cname == "LCCC":
                return "LINK_FAIL", f"LCCC link failed: {r3.stderr[:100]}"
            return "SKIP", "GCC link failed"

    # Run both and compare
    gcc_r = subprocess.run([str(tmp / "multi_GCC")], capture_output=True, text=True, timeout=10)
    lccc_r = subprocess.run([str(tmp / "multi_LCCC")], capture_output=True, text=True, timeout=10)

    if gcc_r.stdout.strip() == lccc_r.stdout.strip():
        return "PASS", ""
    return "MISMATCH", f"GCC: {gcc_r.stdout.strip()}\nLCCC: {lccc_r.stdout.strip()}"

def main():
    p = argparse.ArgumentParser(description="LCCC correctness test suite")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--filter", help="only run tests matching this substring")
    args = p.parse_args()

    if not LCCC.exists():
        print(c(RED, f"ERROR: LCCC not found at {LCCC}"))
        sys.exit(1)

    print(c(BOLD, "\nLCCC Correctness Test Suite"))
    print(f"  LCCC: {LCCC}")
    print(f"  GCC include: {GCC_INC}")
    print()

    pass_count = 0
    fail_count = 0
    skip_count = 0
    compile_fail_count = 0
    failures = []

    with tempfile.TemporaryDirectory(prefix="lccc_test_") as tmpdir:
        for name, src, extra_flags, expected_exit in TESTS:
            if args.filter and args.filter not in name:
                continue

            status, detail = run_test(name, src, extra_flags, args.verbose, tmpdir)

            if status == "PASS":
                print(f"  {c(GRN, 'PASS')}  {name}")
                pass_count += 1
            elif status == "SKIP":
                print(f"  {c(YLW, 'SKIP')}  {name}: {detail}")
                skip_count += 1
            elif status == "COMPILE_FAIL":
                print(f"  {c(RED, 'FAIL')}  {name} (compile failed)")
                if args.verbose:
                    print(f"         {detail}")
                compile_fail_count += 1
                failures.append((name, "COMPILE_FAIL", detail))
            elif status == "MISMATCH":
                print(f"  {c(RED, 'FAIL')}  {name} (output mismatch)")
                if args.verbose:
                    for line in detail.split('\n'):
                        print(f"         {line}")
                fail_count += 1
                failures.append((name, "MISMATCH", detail))
            else:
                print(f"  {c(RED, 'FAIL')}  {name} ({status})")
                if args.verbose and detail:
                    print(f"         {detail}")
                fail_count += 1
                failures.append((name, status, detail))

        # Multi-file test
        if not args.filter or "multi_file" in (args.filter or ""):
            status, detail = run_multi_file_test(args.verbose, tmpdir)
            name = "multi_file_compile_link"
            if status == "PASS":
                print(f"  {c(GRN, 'PASS')}  {name}")
                pass_count += 1
            elif status == "SKIP":
                print(f"  {c(YLW, 'SKIP')}  {name}: {detail}")
                skip_count += 1
            else:
                print(f"  {c(RED, 'FAIL')}  {name} ({status})")
                if args.verbose and detail:
                    print(f"         {detail}")
                fail_count += 1
                failures.append((name, status, detail))

    total = pass_count + fail_count + compile_fail_count + skip_count
    print(f"\n{'='*60}")
    print(f"  Results: {c(GRN, str(pass_count))} passed, "
          f"{c(RED, str(fail_count + compile_fail_count))} failed "
          f"({compile_fail_count} compile, {fail_count} runtime), "
          f"{c(YLW, str(skip_count))} skipped — {total} total")

    if failures:
        print(f"\n  {c(RED, 'Failed tests:')}")
        for name, status, detail in failures:
            print(f"    {name}: {status}")
            if detail:
                for line in detail.split('\n')[:3]:
                    print(f"      {line}")

    sys.exit(1 if (fail_count + compile_fail_count) > 0 else 0)

if __name__ == "__main__":
    main()
