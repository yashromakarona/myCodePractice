#include <stdio.h>

void exchange (int *, int *);

int main() {
    int x, y;
    x = 100;
    y = 200;
    
    printf("Before exchange: x = %d, y = %d\n", x, y);
    exchange(&x, &y);
    printf("After exchange: x = %d, y = %d\n", x, y);

    return 0;
}

void exchange(int *a, int *b) {
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
