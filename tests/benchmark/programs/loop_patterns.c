// Loop optimization patterns (unrolling, strength reduction, LICM targets)
#include <stdio.h>

#define SIZE 10000000

static int array[SIZE];

// Simple reduction - should vectorize
static long sum_array(int *arr, int n) {
    long s = 0;
    for (int i = 0; i < n; i++)
        s += arr[i];
    return s;
}

// Conditional reduction
static long sum_positive(int *arr, int n) {
    long s = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] > 0) s += arr[i];
    return s;
}

// Max element
static int find_max(int *arr, int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx) mx = arr[i];
    return mx;
}

// Array copy with transform
static void scale_add(int *dst, const int *src, int n, int scale, int offset) {
    for (int i = 0; i < n; i++)
        dst[i] = src[i] * scale + offset;
}

// Dot product (int)
static long dot_product(const int *a, const int *b, int n) {
    long s = 0;
    for (int i = 0; i < n; i++)
        s += (long)a[i] * b[i];
    return s;
}

// Prefix sum
static void prefix_sum(int *arr, int n) {
    for (int i = 1; i < n; i++)
        arr[i] += arr[i-1];
}

int main(void) {
    unsigned int seed = 42;
    for (int i = 0; i < SIZE; i++) {
        seed = seed * 1664525u + 1013904223u;
        array[i] = (int)(seed >> 1) - 1000000000;
    }

    long s1 = sum_array(array, SIZE);
    long s2 = sum_positive(array, SIZE);
    int mx = find_max(array, SIZE);

    static int buf[SIZE];
    scale_add(buf, array, SIZE, 3, 7);
    long s3 = sum_array(buf, SIZE);

    long dp = dot_product(array, buf, SIZE / 10);

    // Restore array for prefix sum test
    seed = 42;
    for (int i = 0; i < 10000; i++) {
        seed = seed * 1664525u + 1013904223u;
        array[i] = (int)(seed % 100);
    }
    prefix_sum(array, 10000);

    printf("sum=%ld pos=%ld max=%d scaled=%ld dot=%ld prefix=%d\n",
           s1, s2, mx, s3, dp, array[9999]);
    return 0;
}
