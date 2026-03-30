#include <stdio.h>

int main(void) {
    float a, b;
    float c, d;

    a = 6;
    b = 10;

    c = b / a;
    // The modulo operator (%) requires integer operands.
    // We cast them to int to resolve the build error.
    d = (int)b % (int)a;

    printf("c = %f d = %f \n", c, d);
    return 0;
}
