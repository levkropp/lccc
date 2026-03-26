#!/bin/bash
# LCCC vs GCC Compatibility Test Suite
# Tests real-world C patterns from simple to complex

set -e
GCC_INC="-I$(gcc -print-file-name=include)"
LCCC="./target/release/lccc $GCC_INC -O2"
GCC="gcc -O2"
PASS=0
FAIL=0
SKIP=0
TOTAL=0

test_program() {
    local name="$1"
    local src="$2"
    local expected="$3"
    TOTAL=$((TOTAL + 1))

    echo "$src" > /tmp/lccc_test.c

    # Compile with GCC (reference)
    if ! $GCC -o /tmp/lccc_test_gcc /tmp/lccc_test.c -lm 2>/dev/null; then
        echo "SKIP  $name (GCC compile failed)"
        SKIP=$((SKIP + 1))
        return
    fi

    # Get GCC output
    local gcc_out=$(/tmp/lccc_test_gcc 2>&1 || true)

    # Compile with LCCC
    if ! $LCCC -o /tmp/lccc_test_lccc /tmp/lccc_test.c -lm 2>/dev/null; then
        echo "FAIL  $name (LCCC compile failed)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Get LCCC output
    local lccc_out=$(/tmp/lccc_test_lccc 2>&1 || true)

    if [ "$gcc_out" = "$lccc_out" ]; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        echo "  GCC:  $gcc_out"
        echo "  LCCC: $lccc_out"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== LCCC Compatibility Tests ==="
echo ""

# --- Level 1: Basic C ---
echo "--- Level 1: Basic C ---"

test_program "hello world" '
#include <stdio.h>
int main() { printf("hello world\n"); return 0; }
'

test_program "arithmetic" '
#include <stdio.h>
int main() { printf("%d %d %d\n", 2+3, 10*7, 100/3); return 0; }
'

test_program "string ops" '
#include <stdio.h>
#include <string.h>
int main() {
    char buf[100] = "hello";
    strcat(buf, " world");
    printf("%s len=%zu\n", buf, strlen(buf));
    return 0;
}
'

test_program "for loop" '
#include <stdio.h>
int main() {
    int sum = 0;
    for (int i = 1; i <= 100; i++) sum += i;
    printf("sum=%d\n", sum);
    return 0;
}
'

test_program "while + break" '
#include <stdio.h>
int main() {
    int n = 1;
    while (n < 1000) { n *= 2; if (n > 500) break; }
    printf("n=%d\n", n);
    return 0;
}
'

# --- Level 2: Functions and Pointers ---
echo ""
echo "--- Level 2: Functions and Pointers ---"

test_program "recursion" '
#include <stdio.h>
int fib(int n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }
int main() { printf("fib(10)=%d\n", fib(10)); return 0; }
'

test_program "function pointers" '
#include <stdio.h>
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }
int main() {
    int (*ops[])(int,int) = {add, mul};
    printf("%d %d\n", ops[0](3,4), ops[1](3,4));
    return 0;
}
'

test_program "pointer arithmetic" '
#include <stdio.h>
int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int *p = arr + 2;
    printf("%d %d %d\n", *p, *(p-1), *(p+2));
    return 0;
}
'

test_program "malloc/free" '
#include <stdio.h>
#include <stdlib.h>
int main() {
    int *p = malloc(10 * sizeof(int));
    for (int i = 0; i < 10; i++) p[i] = i * i;
    int sum = 0;
    for (int i = 0; i < 10; i++) sum += p[i];
    free(p);
    printf("sum=%d\n", sum);
    return 0;
}
'

# --- Level 3: Structs ---
echo ""
echo "--- Level 3: Structs ---"

test_program "basic struct" '
#include <stdio.h>
typedef struct { int x, y; } Point;
int main() {
    Point p = {3, 4};
    printf("(%d,%d)\n", p.x, p.y);
    return 0;
}
'

test_program "struct array" '
#include <stdio.h>
typedef struct { int x, y; } Point;
int main() {
    Point pts[] = {{1,2}, {3,4}, {5,6}};
    int sum = 0;
    for (int i = 0; i < 3; i++) sum += pts[i].x + pts[i].y;
    printf("sum=%d\n", sum);
    return 0;
}
'

test_program "struct with function pointer" '
#include <stdio.h>
typedef struct { const char *name; int (*func)(int); } Op;
static int dbl(int x) { return x * 2; }
static int sqr(int x) { return x * x; }
int main() {
    Op ops[] = {{"dbl", dbl}, {"sqr", sqr}, {0, 0}};
    for (int i = 0; ops[i].name; i++)
        printf("%s(%d)=%d\n", ops[i].name, 5, ops[i].func(5));
    return 0;
}
'

test_program "linked list" '
#include <stdio.h>
#include <stdlib.h>
typedef struct Node { int val; struct Node *next; } Node;
int main() {
    Node *head = NULL;
    for (int i = 5; i >= 1; i--) {
        Node *n = malloc(sizeof(Node));
        n->val = i; n->next = head; head = n;
    }
    int sum = 0;
    for (Node *p = head; p; p = p->next) sum += p->val;
    printf("sum=%d\n", sum);
    // leak is ok for test
    return 0;
}
'

# --- Level 4: Complex patterns ---
echo ""
echo "--- Level 4: Complex patterns ---"

test_program "switch statement" '
#include <stdio.h>
int classify(int n) {
    switch (n % 4) {
        case 0: return 100;
        case 1: return 200;
        case 2: return 300;
        default: return 400;
    }
}
int main() {
    int sum = 0;
    for (int i = 0; i < 20; i++) sum += classify(i);
    printf("sum=%d\n", sum);
    return 0;
}
'

test_program "qsort callback" '
#include <stdio.h>
#include <stdlib.h>
int cmp(const void *a, const void *b) { return *(int*)a - *(int*)b; }
int main() {
    int arr[] = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
    qsort(arr, 10, sizeof(int), cmp);
    for (int i = 0; i < 10; i++) printf("%d ", arr[i]);
    printf("\n");
    return 0;
}
'

test_program "varargs" '
#include <stdio.h>
#include <stdarg.h>
int sum_n(int n, ...) {
    va_list ap;
    va_start(ap, n);
    int s = 0;
    for (int i = 0; i < n; i++) s += va_arg(ap, int);
    va_end(ap);
    return s;
}
int main() { printf("sum=%d\n", sum_n(4, 10, 20, 30, 40)); return 0; }
'

test_program "setjmp/longjmp" '
#include <stdio.h>
#include <setjmp.h>
jmp_buf env;
void bail(int code) { longjmp(env, code); }
int main() {
    int v = setjmp(env);
    if (v == 0) { bail(42); }
    printf("caught %d\n", v);
    return 0;
}
'

test_program "floating point" '
#include <stdio.h>
#include <math.h>
int main() {
    double x = 3.14159265;
    printf("sin=%.4f cos=%.4f sqrt2=%.4f\n", sin(x), cos(x), sqrt(2.0));
    return 0;
}
'

# --- Summary ---
echo ""
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped (of $TOTAL) ==="
