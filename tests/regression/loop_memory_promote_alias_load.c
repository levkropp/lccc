extern int printf(const char *, ...);
extern void *memset(void *, int, unsigned long);
extern char *strcpy(char *, const char *);

/* Regression: loop_memory_promote must not promote a loop-carried store when
 * another loop load reaches the same address through a different pointer value
 * (e.g. duplicate GEPs hoisted by LICM). The unpromoted loads kept reading
 * stale memory, so the decrement never became visible and the loop spun
 * forever (sqlite3FpDecode trailing-zero trim, sqlite workload hang).
 * Correct output: n=0 (three trailing zeros trimmed, then 'x' stops it). */
typedef struct {
    int flags;
    int n;
    char *z;
    char zBuf[24];
} FpD;

static void trim_zeros(FpD *p) {
    p->z = &p->zBuf[1];
    while (p->n > 0 && p->z[p->n - 1] == '0') {
        p->n--;
    }
}

int main(void) {
    FpD p;
    memset(&p, 0, sizeof(p));
    p.n = 3;
    strcpy(p.zBuf, "x000");
    trim_zeros(&p);
    printf("n=%d\n", p.n);
    return 0;
}
