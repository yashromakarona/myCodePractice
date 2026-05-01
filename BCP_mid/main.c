#include <stdio.h>

int main(void) {
    float a, b;

    printf("Enter a value of a: ");
    scanf("%f", &a);

    printf("Enter a value of b: ");
    scanf("%f", &b);

    if (b == 0)
        printf("We can't divide by 0\n");
    else
        printf("%.3f / %.3f = %.3f\n", a, b, a / b);
}




