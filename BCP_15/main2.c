#include <stdio.h>

int main () {
    int x, y;
    int *ptr;
    x = 10;
    ptr = &x;
    y = *ptr;
    printf ("x = %d\n", x);
    printf("%d is stored at addr %u\n", x, &x);
    printf("%d is stored at addr %u\n", *&x, &x);
    printf("%d is stored at addr %u\n", y, &y);
    printf("%u is stored at addr %u\n", ptr, &ptr);
    printf("%d is stored at addr %u\n", *ptr, ptr);

    return 0;
}
