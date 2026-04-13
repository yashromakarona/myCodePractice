#include <stdio.h>

int main(void) {
    int years, dep_rate, or_price, dep_price;

    printf("\nEnter original price :");
    scanf("%u", &or_price);
    printf("\nEnter number of years used :");
    scanf("%u", &years);

    switch (years)
    {
        case 1:
            dep_rate = 8;
            break;
        case 2:
            dep_rate = 20;
            break;
        case 3:
            dep_rate = 35;
            break;
        case 4:
            dep_rate = 50;
            break;
        case 5:
            dep_rate = 65;
            break;
        case 6:
            dep_rate = 80;
            break;
        default:
            dep_rate = 100;
    }
    dep_price = (int)(or_price - (dep_rate/100.0)*or_price);
    printf("\nThe depreciated price = %d", dep_price);
}