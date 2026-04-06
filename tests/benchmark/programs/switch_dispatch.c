// Switch/dispatch benchmark (branch prediction, jump tables)
#include <stdio.h>

#define N 50000000

static int dispatch(int op, int a, int b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a ^ b;
        case 4: return a | b;
        case 5: return a & b;
        case 6: return a << (b & 7);
        case 7: return a >> (b & 7);
        case 8: return ~a + b;
        case 9: return a - ~b;
        case 10: return (a + b) * 3;
        case 11: return (a - b) * 5;
        case 12: return (a ^ b) + 1;
        case 13: return (a | b) - 1;
        case 14: return (a & b) + 2;
        case 15: return a + b + 1;
        default: return 0;
    }
}

int main(void) {
    long sum = 0;
    unsigned int seed = 777;
    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        int op = (seed >> 16) & 15;
        sum += dispatch(op, i, (int)(seed & 0xffff));
    }
    printf("switch_dispatch sum: %ld\n", sum);
    return 0;
}
