//
// Created by 최태우 on 2026. 5. 11..
//

#include <stdio.h>
#include <string.h>

void main() {
    char str[30];
    int vow=0,cons=0,i=0;
    printf("Enter a string: ");
    scanf("%s", str);

    while (str[i] != '\0') {
        if ((str[i] >= 'a' && str[i] <= 'z') || (str[i] >= 'A' && str[i] <= 'Z')) {
            if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' || str[i] == 'u' ||
                str[i] == 'A' || str[i] == 'E' || str[i] == 'I' || str[i] == 'O' || str[i] == 'U') {
                vow++;
            } else {
                cons++;
            }
        }
        i++;
    }

    printf("Vowels: %d\n", vow);
    printf("Consonants: %d\n", cons);
}
