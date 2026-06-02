#include <stdio.h>
int main() {
    int amount;
    int denom[] = {50000, 10000, 5000, 1000, 500, 100, 50, 10, 5, 1};
    printf("Enter amount in KRW: ");
    scanf("%d", &amount);
    for (int i = 0; i < 10; i++) {
        if (amount >= denom[i]) {
            printf("%5d KRW: %d\n", denom[i], amount / denom[i]);
            amount %= denom[i];
        }
    }
    return 0;
}