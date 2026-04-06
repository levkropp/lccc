// Spectral norm benchmark (FP-heavy, array access patterns)
#include <stdio.h>
#include <math.h>

#define N 2000

static double A(int i, int j) {
    return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1);
}

static void mul_Av(int n, const double *v, double *out) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++)
            sum += A(i, j) * v[j];
        out[i] = sum;
    }
}

static void mul_Atv(int n, const double *v, double *out) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++)
            sum += A(j, i) * v[j];
        out[i] = sum;
    }
}

static void mul_AtAv(int n, const double *v, double *out) {
    double tmp[N];
    mul_Av(n, v, tmp);
    mul_Atv(n, tmp, out);
}

int main(void) {
    int n = N;
    double u[N], v[N];
    for (int i = 0; i < n; i++) u[i] = 1.0;

    for (int i = 0; i < 10; i++) {
        mul_AtAv(n, u, v);
        mul_AtAv(n, v, u);
    }

    double vBv = 0, vv = 0;
    for (int i = 0; i < n; i++) {
        vBv += u[i] * v[i];
        vv += v[i] * v[i];
    }
    printf("%.9f\n", sqrt(vBv / vv));
    return 0;
}
