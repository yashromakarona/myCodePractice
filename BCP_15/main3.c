#include <stdio.h>

int main() {
    int a, b, *p1, *p2, x, y, z;
    a = 12;
    b = 4;
    p1 = &a;
    p2 = &b;

    x = *p1 * *p2 - 6;
    y = 4 * - *p2 / *p1 + 10;

    printf("a = %d, b = %d\n", a, b);
    printf("p1 = %u, p2 = %u\n", p1, p2);
    printf("x = %d, y = %d\n", x, y);

    *p1 = *p1 + 3;
    *p2 = *p2 + 2;
    z = *p1 * *p2 - 6;
    printf("a = %d, b = %d, z = %d\n", a, b, z);

    return 0;
}
