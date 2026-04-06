// Benchmark 5: Sieve of Eratosthenes (memory writes + branching)
#include <stdio.h>
#include <string.h>

#define N 10000000

static char sieve[N+1];

int count_primes(void) {
    memset(sieve, 1, sizeof(sieve));
    sieve[0] = sieve[1] = 0;
    for (int i = 2; i * i <= N; i++)
        if (sieve[i])
            for (int j = i*i; j <= N; j += i)
                sieve[j] = 0;
    int count = 0;
    for (int i = 2; i <= N; i++)
        if (sieve[i]) count++;
    return count;
}

int main(void) {
    volatile int result = count_primes();
    printf("primes up to %d: %d\n", N, result);
    return 0;
}
