// Mandelbrot set computation (FP-heavy inner loop, branching)
#include <stdio.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 50

int main(void) {
    int total = 0;
    for (int y = 0; y < HEIGHT; y++) {
        double ci = 2.0 * y / HEIGHT - 1.0;
        for (int x = 0; x < WIDTH; x++) {
            double cr = 2.0 * x / WIDTH - 1.5;
            double zr = 0.0, zi = 0.0;
            int i;
            for (i = 0; i < MAX_ITER; i++) {
                double tr = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = tr;
                if (zr * zr + zi * zi > 4.0) break;
            }
            total += i;
        }
    }
    printf("mandelbrot total iterations: %d\n", total);
    return 0;
}
