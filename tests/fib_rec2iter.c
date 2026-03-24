// End-to-end test for binary recursion-to-iteration optimization.
// Verifies correctness across multiple Fibonacci inputs.
// When compiled with LCCC, the rec2iter pass should transform this
// from O(2^n) to O(n), making it run near-instantly.
#include <stdio.h>

long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    // Test correctness across a range of inputs
    long expected[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
    int pass = 1;
    for (int i = 0; i <= 12; i++) {
        long result = fib(i);
        if (result != expected[i]) {
            printf("FAIL: fib(%d) = %ld, expected %ld\n", i, result, expected[i]);
            pass = 0;
        }
    }

    // Test larger values (only feasible if rec2iter fires — O(n) vs O(2^n))
    long fib40 = fib(40);
    if (fib40 != 102334155L) {
        printf("FAIL: fib(40) = %ld, expected 102334155\n", fib40);
        pass = 0;
    }

    long fib50 = fib(50);
    if (fib50 != 12586269025L) {
        printf("FAIL: fib(50) = %ld, expected 12586269025\n", fib50);
        pass = 0;
    }

    // fib(90) would take billions of years without rec2iter
    long fib90 = fib(90);
    if (fib90 != 2880067194370816120L) {
        printf("FAIL: fib(90) = %ld, expected 2880067194370816120\n", fib90);
        pass = 0;
    }

    if (pass) {
        printf("ALL PASS\n");
    }
    return pass ? 0 : 1;
}
