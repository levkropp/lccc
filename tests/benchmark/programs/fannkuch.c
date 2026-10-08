// Fannkuch-Redux benchmark (array permutations, integer-heavy)
#include <stdio.h>

static int perm[16], perm1[16], count[16], maxflips, checksum;

static void fannkuch(int n) {
    maxflips = 0;
    checksum = 0;

    for (int i = 0; i < n; i++) perm1[i] = i;

    int r = n;
    int nperm = 0;

    for (;;) {
        while (r > 1) { count[r - 1] = r; r--; }

        for (int i = 0; i < n; i++) perm[i] = perm1[i];

        int flips = 0;
        int k;
        while ((k = perm[0]) != 0) {
            int k2 = (k + 1) >> 1;
            for (int i = 0; i < k2; i++) {
                int t = perm[i];
                perm[i] = perm[k - i];
                perm[k - i] = t;
            }
            flips++;
        }

        if (flips > maxflips) maxflips = flips;
        checksum += (nperm & 1) ? -flips : flips;
        nperm++;

        for (;;) {
            if (r == n) return;
            int p0 = perm1[0];
            for (int i = 0; i < r; i++) perm1[i] = perm1[i + 1];
            perm1[r] = p0;
            count[r]--;
            if (count[r] > 0) break;
            r++;
        }
    }
}

int main(void) {
    int n = 11;
    fannkuch(n);
    printf("%d\nPfannkuchen(%d) = %d\n", checksum, n, maxflips);
    return 0;
}
