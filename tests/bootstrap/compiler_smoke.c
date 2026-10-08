struct Pair {
    long first;
    long second;
};

static long dot(const struct Pair *pairs, int count) {
    long result = 0;
    for (int i = 0; i < count; i++)
        result += pairs[i].first * pairs[i].second;
    return result;
}

int main(void) {
    struct Pair pairs[] = {{2, 3}, {4, 5}, {6, 7}};
    return dot(pairs, 3) == 68 ? 0 : 1;
}
