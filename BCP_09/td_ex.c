//
// Created by 최태우 on 2026. 5. 4..//
#include <stdio.h>

int main() {
    int row, colum, product[3][3];
    int i, j;
    printf("MULTIPLICATION TABLE\n\n");
    printf(" ");
    for (j = 1; j <= 3; j++)
        printf("%4d", j);
    printf("\n");
    printf("------------------------------\n");
    for (i = 0; i < 3; i++) {
        row = i + 1;
        printf("%2d |", row);
        for (j = 1; j <= 3; j++) {
            colum = j;
            product[i][j - 1] = row * colum;
            printf("%4d", product[i][j - 1]);
        }
        printf("\n");
    }
    return 0;
}
