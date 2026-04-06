// Benchmark 6: Tail-recursive accumulator sum (TCE demonstration)
// sum(n, acc) is a self-recursive tail call -> converted to a loop by TCE.
// Without TCE: 10M stack frames. With TCE: a single tight loop.
#include <stdio.h>

static long sum(int n, long acc) {
    if (n <= 0) return acc;
    return sum(n - 1, acc + n);
}

int main(void) {
    volatile long result = sum(10000000, 0);
    printf("sum(10000000) = %ld\n", result);
    return 0;
}
