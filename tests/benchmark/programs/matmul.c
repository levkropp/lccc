// Benchmark 3: Matrix multiplication (memory/cache + arithmetic)
#include <stdio.h>

#define N 256

static double A[N][N], B[N][N], C[N][N];

void matmul(void) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];
}

int main(void) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / N;
            B[i][j] = (double)(i * j + 1) / N;
        }
    matmul();
    volatile double result = C[N/2][N/2];
    printf("matmul C[128][128] = %.4f\n", result);
    return 0;
}
