// Benchmark 1: Arithmetic-heavy loop with many local variables
// This stresses register allocation — 32 local ints
#include <stdio.h>

int arith_loop(int n) {
    int a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8;
    int i = 9, j = 10, k = 11, l = 12, m = 13, o = 14, p = 15, q = 16;
    int r = 17, s = 18, t = 19, u = 20, v = 21, w = 22, x = 23, y = 24;
    int z0 = 25, z1 = 26, z2 = 27, z3 = 28, z4 = 29, z5 = 30, z6 = 31, z7 = 32;

    for (int iter = 0; iter < n; iter++) {
        a += b * c; b += c * d; c += d * e; d += e * f;
        e += f * g; f += g * h; g += h * i; h += i * j;
        i += j * k; j += k * l; k += l * m; l += m * o;
        m += o * p; o += p * q; p += q * r; q += r * s;
        r += s * t; s += t * u; t += u * v; u += v * w;
        v += w * x; w += x * y; x += y * z0; y += z0 * z1;
        z0 += z1 * z2; z1 += z2 * z3; z2 += z3 * z4; z3 += z4 * z5;
        z4 += z5 * z6; z5 += z6 * z7; z6 += z7 * a; z7 += a * b;
    }
    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h ^ i ^ j ^ k ^ l ^
           m ^ o ^ p ^ q ^ r ^ s ^ t ^ u ^ v ^ w ^ x ^ y ^
           z0 ^ z1 ^ z2 ^ z3 ^ z4 ^ z5 ^ z6 ^ z7;
}

int main(void) {
    volatile int result = arith_loop(10000000);
    printf("arith_loop result: %d\n", result);
    return 0;
}
