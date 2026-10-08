// Test suite for Phase 9b: IVSR integration with indexed addressing
//
// Tests indexed addressing detection for IVSR-transformed loop patterns:
// - Simple array sum (single array)
// - Multiple arrays (a[i] + b[i] -> c[i])
// - Non-power-of-2 stride (should fall back to pointer increment)

// Test 1: Simple array sum
// Expected: movsd (%rdi,%rax,8), %xmm1
double sum_array(double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Test 2: Multiple arrays
// Expected: movsd (%rdi,%rax,8), %xmm0
//           addsd (%rsi,%rax,8), %xmm0
//           movsd %xmm0, (%rdx,%rax,8)
void add_arrays(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Test 3: Integer array
// Expected: movq (%rdi,%rax,8), %rbx (or similar)
long sum_longs(long *arr, int n) {
    long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Test 4: Smaller element size (int = 4 bytes, scale=4)
// Expected: movl (%rdi,%rax,4), %ebx
int sum_ints(int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Test 5: Struct with non-power-of-2 size
// Expected: Fallback to pointer increment (stride=24 not valid SIB scale)
struct Point {
    double x, y, z;  // 24 bytes total
};

double sum_points_x(struct Point *pts, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += pts[i].x;
    }
    return sum;
}

// Test 6: Array write pattern
// Expected: movsd %xmm0, (%rdi,%rax,8)
void fill_array(double *arr, int n, double value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

// Test 7: Read-modify-write
// Expected: Both load and store use indexed addressing
void scale_array(double *arr, int n, double factor) {
    for (int i = 0; i < n; i++) {
        arr[i] *= factor;
    }
}
