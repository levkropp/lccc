static int sum(const int *values, int count) {
    int result = 0;
    for (int i = 0; i < count; i++)
        result += values[i];
    return result;
}

int main(void) {
    int values[] = {10, 20, 30, 40, 50};
    int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};

    if (sum(values, 5) != 150 || sum(values + 2, 3) != 120)
        return 1;
    if ((values + 4) - (values + 1) != 3)
        return 2;
    if (matrix[1][2] != 6)
        return 3;
    return 0;
}
