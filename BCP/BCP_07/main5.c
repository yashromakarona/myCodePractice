#include <stdio.h>

#define LOOP 100
#define ACCURACY 0.0001

int main() {
    int n;
    float x, term, sum;
    printf("Input value of x: ");
    scanf("%f", &x);
    sum = 0;
    for (term = 1, n = 1; n <= LOOP; ++n) {
        sum += term;
        if (term <= ACCURACY)
            goto output;
        term *= x;

    }
    printf("\nFINAL VALUE OF N IS NOT SUFFICIENT\n");
    printf("TO ACHIEVE DESIRED ACCRACY\n");
    goto end;
    output:
    printf("EXIT FROM LOOP\n");
    printf("SUM = %f; N0.of terms = %d\n", sum, n);
    end:
    ;
}