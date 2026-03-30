#include <stdio.h>

int main(void) {
    int a, b;
    float c, d;

    a = 6;
    b = 10;

    c = b / a;
    d = b % a;

    printf("c = %5.4f d = %f \n", c, d);
}
