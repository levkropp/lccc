// String processing benchmark (byte-level ops, branching, memcpy patterns)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define NSTRINGS 100000
#define MAX_LEN 200

static char strings[NSTRINGS][MAX_LEN];

static unsigned int lcg(unsigned int *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

// Simple hand-written strstr for benchmarking
static const char *my_strstr(const char *haystack, const char *needle) {
    if (!*needle) return haystack;
    int nlen = 0;
    for (const char *p = needle; *p; p++) nlen++;
    for (const char *h = haystack; *h; h++) {
        const char *a = h, *b = needle;
        while (*a && *b && *a == *b) { a++; b++; }
        if (!*b) return h;
    }
    return NULL;
}

int main(void) {
    unsigned int seed = 42;

    // Generate random strings
    for (int i = 0; i < NSTRINGS; i++) {
        int len = 10 + (lcg(&seed) % (MAX_LEN - 20));
        for (int j = 0; j < len; j++)
            strings[i][j] = 'a' + (lcg(&seed) % 26);
        strings[i][len] = '\0';
    }

    // Benchmark: strlen all strings
    long total_len = 0;
    for (int rep = 0; rep < 50; rep++)
        for (int i = 0; i < NSTRINGS; i++)
            total_len += strlen(strings[i]);

    // Benchmark: strcmp pairs
    long cmp_sum = 0;
    for (int i = 0; i < NSTRINGS - 1; i++)
        cmp_sum += strcmp(strings[i], strings[i + 1]);

    // Benchmark: strstr search
    long found = 0;
    char needle[4] = "abc";
    for (int i = 0; i < NSTRINGS; i++)
        if (my_strstr(strings[i], needle)) found++;

    // Benchmark: memcpy
    char buf[MAX_LEN];
    long copy_sum = 0;
    for (int rep = 0; rep < 50; rep++) {
        for (int i = 0; i < NSTRINGS; i++) {
            int len = strlen(strings[i]);
            memcpy(buf, strings[i], len + 1);
            copy_sum += buf[0];
        }
    }

    printf("strlen total: %ld, cmp_sum: %ld, found: %ld, copy_sum: %ld\n",
           total_len, cmp_sum, found, copy_sum);
    return 0;
}
