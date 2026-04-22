#include <stdio.h>
#include <stdlib.h> // system()

int main(void) {
    int num, i, y, x=40;
    system("clear");
    printf("\nEnter a number for \ngenerating the pyramid:\n");
    scanf("%d", &num);

    for (y = 0; y <= num; y++) {
        for (i = 1; i <= x - y; i++) {
            printf(" ");
        }
        // y부터 0까지 감소하며 출력
        for (i = y; i >= 0; i--) {
            printf("%d", i);
        }
        // 1부터 y까지 증가하며 출력
        for (i = 1; i <= y; i++) {
            printf("%d", i);
        }
        printf("\n");
    }
    return 0;
}
