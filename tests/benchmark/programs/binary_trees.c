// Binary trees benchmark (malloc/free heavy, recursion, pointer chasing)
#include <stdio.h>
#include <stdlib.h>

typedef struct Node { struct Node *left, *right; } Node;

static Node *make(int depth) {
    Node *n = (Node *)malloc(sizeof(Node));
    if (depth > 0) {
        n->left = make(depth - 1);
        n->right = make(depth - 1);
    } else {
        n->left = n->right = NULL;
    }
    return n;
}

static int check(Node *n) {
    if (n->left == NULL) return 1;
    return 1 + check(n->left) + check(n->right);
}

static void destroy(Node *n) {
    if (n->left) { destroy(n->left); destroy(n->right); }
    free(n);
}

int main(void) {
    int min_depth = 4;
    int max_depth = 18;
    if (max_depth < min_depth + 2) max_depth = min_depth + 2;

    // stretch tree
    int stretch = max_depth + 1;
    Node *t = make(stretch);
    printf("stretch tree of depth %d\t check: %d\n", stretch, check(t));
    destroy(t);

    // long lived tree
    Node *ll = make(max_depth);

    for (int d = min_depth; d <= max_depth; d += 2) {
        int iters = 1;
        for (int i = 0; i < max_depth - d + min_depth; i++) iters *= 2;
        int total = 0;
        for (int i = 0; i < iters; i++) {
            Node *a = make(d);
            total += check(a);
            destroy(a);
        }
        printf("%d\t trees of depth %d\t check: %d\n", iters, d, total);
    }
    printf("long lived tree of depth %d\t check: %d\n", max_depth, check(ll));
    destroy(ll);
    return 0;
}
