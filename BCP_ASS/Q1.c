#include <stdio.h>

int main(void) {
    int num, sum = 0;

    printf("Enter any digits: ");
    scanf("%d", &num);

    if (num < 0) {
        num = -num;
    }

    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }

    printf("The sum of %d digits is: %d\n", num, sum);

    return 0;
}