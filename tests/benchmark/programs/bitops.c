// Bit manipulation benchmark (popcount, clz, bit twiddling)
#include <stdio.h>

#define N 50000000

static int popcount32(unsigned int x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0f0f0f0f;
    return (x * 0x01010101) >> 24;
}

static int clz32(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8;  x <<= 8;  }
    if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4;  }
    if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2;  }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}

static unsigned int reverse_bits(unsigned int x) {
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}

static unsigned int next_power_of_2(unsigned int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    v++;
    return v;
}

int main(void) {
    unsigned int seed = 0xDEADBEEF;
    long pop_sum = 0, clz_sum = 0, rev_sum = 0, pow2_sum = 0;

    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        pop_sum += popcount32(seed);
        clz_sum += clz32(seed);
        rev_sum += reverse_bits(seed) & 0xFF;
        pow2_sum += next_power_of_2(seed & 0xFFFF);
    }

    printf("pop=%ld clz=%ld rev=%ld pow2=%ld\n", pop_sum, clz_sum, rev_sum, pow2_sum);
    return 0;
}
