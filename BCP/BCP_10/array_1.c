//
// Created by 최태우 on 2026. 5. 11..
//
#include <stdio.h>

int main() {
    char word1[40], word2[40], word3[40], word4[40];

    printf("Enter text: \n");
    scanf("%s %s",word1, word2);
    scanf("%s",word3);
    scanf("%s",word4);

    printf("\nword1 = %s\nword2 = %s\nword3 = %s\nword4 = %s\n", word1, word2, word3, word4);

    return 0;
}
