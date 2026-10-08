extern int printf(const char *, ...);

static int counter(void) {
    static int value;
    return ++value;
}

int main(void) {
    for (int i = 0; i < 5; i++)
        printf("%d ", counter());
    return 0;
}
