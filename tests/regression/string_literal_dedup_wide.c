int main(void) {
    const int *wide_a = L"same";
    const int *wide_b = L"same";
    const unsigned short *utf16_a = u"same";
    const unsigned short *utf16_b = u"same";

    return wide_a != wide_b || utf16_a != utf16_b;
}
