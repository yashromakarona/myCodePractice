//
// Created by 최태우 on 2026. 5. 11..
//

#include <stdio.h>

int main() {
    char country[15] = "United Kingdom";
    printf("\n\n");
    printf("123456789012345\n");
    printf("---------------\n");
    printf("%15s\n", country);
    printf("%5s\n", country);
    printf("%15.6s\n", country);
    printf("%-15.6s\n", country);
    printf("%.3s\n", country);
    printf("%s\n", country);
    printf("---------------\n");

    return 0;
}
