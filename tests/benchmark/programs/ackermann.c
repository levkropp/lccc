// Ackermann function - deep recursion stress test (stack, call overhead)
#include <stdio.h>

static int ackermann(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));
}

int main(void) {
    // ack(3,11) = 16381 - deep enough to stress call overhead
    int result = ackermann(3, 11);
    printf("ackermann(3,11) = %d\n", result);
    return 0;
}
