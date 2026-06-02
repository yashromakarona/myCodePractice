#include <stdio.h>
#include <ctype.h>

int main(void) {

    char alpha;

    printf("Enter a alphabet: ");
    scanf("%c", &alpha);

    if (isalpha(alpha)) {
        alpha = tolower(alpha);

        if (alpha == 'a' || alpha == 'e' || alpha == 'i' || alpha == 'o' || alpha == 'u')
            printf("vowel\n");
        else
            printf("consonant\n");
    }
    else {
        printf("your input is not alphabet");
    }
}