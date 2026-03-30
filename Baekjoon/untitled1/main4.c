#include <stdio.h>

int main(void) {
    int a, b, c, d;

    a = 2;
    b = 2;

    printf("a = %d\t b = %d\n", a, b);

    c = a++;
    d = ++b;

    printf("a = %d\t b = %d\n", a, b);
    printf("c = %d\t d = %d\n", c, d);

    return 0;
}
