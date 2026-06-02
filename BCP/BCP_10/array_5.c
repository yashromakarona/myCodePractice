//
// Created by 최태우 on 2026. 5. 11..
//

#include <stdio.h>

int main() {
    char s1[10], s2[10], s3[10];
    int x, l1, l2, l3;
    printf("\nEnter two string constants \n");
    
    scanf("%s %s", s1, s2);
    
    /* Compare strings s1 and s2 */
    x = 0;
    while (s1[x] == s2[x] && s1[x] != '\0') {
        x++;
    }

    if (s1[x] == '\0' && s2[x] == '\0') {
        printf("\nStrings are equal\n");
    } else {
        printf("\nStrings are not equal\n");
    }

    printf("%s %s\n", s1, s2);

    return 0;
}
