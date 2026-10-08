// Benchmark 2: Recursive Fibonacci (function call overhead)
#include <stdio.h>

long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    volatile long result = fib(40);
    printf("fib(40) = %ld\n", result);
    return 0;
}
