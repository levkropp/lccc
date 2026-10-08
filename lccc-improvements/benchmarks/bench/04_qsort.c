// Benchmark 4: Quicksort (branching + memory access)
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

static int arr[N];

static int cmp(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}

int main(void) {
    unsigned int seed = 42;
    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        arr[i] = (int)(seed & 0x7fffffff);
    }
    qsort(arr, N, sizeof(int), cmp);
    printf("qsort: arr[500000] = %d\n", arr[N/2]);
    return 0;
}
