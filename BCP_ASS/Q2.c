#include <stdio.h>

int main() {
    int age, num = 0;

    for (int i = 0; i < 10; i++) {
        printf("Enter your age: ");
        scanf("%d", &age);

        if (age < 50 || age > 60) {
            continue;
        }
        num++;
    }
    printf("number of people 50 to 60: %d\n", num);
}