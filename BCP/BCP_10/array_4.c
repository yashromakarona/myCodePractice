//
// Created by 최태우 on 2026. 5. 11..
//

#include <stdio.h>

int main() {
    int i, j, k;
    char first_name[10] = "VISWANATH";
    char second_name[10] = "PRATAP";
    char last_name[10] = "SINGH";
    char name[30];

    /* Copy first_name into name */
    for (i = 0; first_name[i] != '\0'; i++) {
        name[i] = first_name[i];
    }
    name[i] = ' ';

    /* Copy second_name into name */
    for (j = 0; second_name[j] != '\0'; j++) {
        name[i + j + 1] = second_name[j];
    }
    name[i + j + 1] = ' ';

    /* Copy last_name into name */
    for (k = 0; last_name[k] != '\0'; k++) {
        name[i + j + k + 2] = last_name[k];
    }
    /* End name with a null char */
    name[i + j + k + 2] = '\0';

    printf("%s\n", name);

    return 0;
}
