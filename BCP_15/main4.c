#include<stdio.h>

int main() {
    int *p, sum, i;
    int x[5] = {5,9,6,3,7};
    i = 0;
    p = x;
    sum = 0;
    printf("Element Value Address\n\n");

    while (i < 5) {
        printf("x[%d] \t %d \t %u\n", i, *p, p);
        sum += *p;
        i++;
        p++;
    }

    printf("\nSum = %d\n", sum);

    return 0;
}
