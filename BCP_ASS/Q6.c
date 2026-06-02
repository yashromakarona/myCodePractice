#include <stdio.h>
int main() {
    char str[] = "123456789";
    for (int i = 1; i <= 5; i++) {
        for (int space = 1; space <= 5 - i; space++) printf("  ");
        for (int j = i; j <= 2 * i - 1; j++) printf("%c ", str[j - 1]);
        for (int j = 2 * i - 2; j >= i; j--) printf("%c ", str[j - 1]);
        printf("\n");
    }
    return 0;
}