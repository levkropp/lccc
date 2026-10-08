// Hash table benchmark (pointer chasing, branching, memory allocation)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 65536
#define NUM_OPS 2000000

typedef struct Entry { unsigned int key; int value; struct Entry *next; } Entry;

static Entry *table[TABLE_SIZE];

static unsigned int hash(unsigned int k) {
    k ^= k >> 16;
    k *= 0x45d9f3b;
    k ^= k >> 16;
    k *= 0x45d9f3b;
    k ^= k >> 16;
    return k & (TABLE_SIZE - 1);
}

static void insert(unsigned int key, int value) {
    unsigned int h = hash(key);
    Entry *e = table[h];
    while (e) {
        if (e->key == key) { e->value = value; return; }
        e = e->next;
    }
    e = (Entry *)malloc(sizeof(Entry));
    e->key = key;
    e->value = value;
    e->next = table[h];
    table[h] = e;
}

static int lookup(unsigned int key) {
    unsigned int h = hash(key);
    Entry *e = table[h];
    while (e) {
        if (e->key == key) return e->value;
        e = e->next;
    }
    return -1;
}

int main(void) {
    unsigned int seed = 12345;
    long sum = 0;

    // Insert phase
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1664525u + 1013904223u;
        insert(seed, i);
    }

    // Lookup phase
    seed = 12345;
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1664525u + 1013904223u;
        sum += lookup(seed);
    }

    // Mixed insert/lookup
    for (int i = 0; i < NUM_OPS; i++) {
        seed = seed * 1664525u + 1013904223u;
        if (i & 1) {
            insert(seed, i);
        } else {
            sum += lookup(seed);
        }
    }

    printf("hash_table sum: %ld\n", sum);
    return 0;
}
