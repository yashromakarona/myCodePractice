#include <stdio.h>
int main() {

    int sum = 0, count = 0;

    printf("%10s %10s\n", "Count", "Number");

    for(int i = 1; i < 100; i++) {
        if(i % 6 == 0 && i % 4 != 0) {
            sum += i;
            count++;
            printf("%10d %10d\n", count, i);
        }
    }
    printf("Sum: %d\n", sum);
    return 0;
}
