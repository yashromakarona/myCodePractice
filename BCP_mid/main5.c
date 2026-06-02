#include <stdio.h>

int main(void) {
    int x = 5;
    int y;

    y = x++ + ++x + x++;

    printf("%d %d",x, y);
}