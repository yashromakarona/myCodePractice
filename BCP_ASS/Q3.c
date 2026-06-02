#include <stdio.h>

int main() {
    int roll_no[5];
    int marks[5][3];
    int total[5] = {0};

    int highest_subject[3] = {0};
    int highest_roll[3] = {0};

    int highest_total = 0;
    int highest_total_roll = 0;

    for (int i = 0; i < 5; i++) {
        printf("Enter Roll No. for student %d: ", i + 1);
        scanf("%d", &roll_no[i]);
        printf("Enter marks for 3 subjects (e.g., 85 90 95): \n");
        scanf("%d %d %d", &marks[i][0], &marks[i][1], &marks[i][2]);

        total[i] = marks[i][0] + marks[i][1] + marks[i][2];
    }

    printf("--- (a) Total marks by each student ---\n");
    for (int i = 0; i < 5; i++) {
        printf("Roll No: %d, Total Marks: %d\n", roll_no[i], total[i]);
    }

    for (int j = 0; j < 3; j++) {
        highest_subject[j] = marks[0][j];
        highest_roll[j] = roll_no[0];

        for (int i = 1; i < 5; i++) {
            if (marks[i][j] > highest_subject[j]) {
                highest_subject[j] = marks[i][j];
                highest_roll[j] = roll_no[i];
            }
        }
    }

    printf("\n--- (b) Highest marks in each subject ---\n");
    for (int j = 0; j < 3; j++) {
        printf("Subject %d: Highest Marks = %d (Roll No: %d)\n", j + 1, highest_subject[j], highest_roll[j]);
    }

    highest_total = total[0];
    highest_total_roll = roll_no[0];
    for (int i = 1; i < 5; i++) {
        if (total[i] > highest_total) {
            highest_total = total[i];
            highest_total_roll = roll_no[i];
        }
    }

    printf("\n--- (c) Student with the highest total marks ---\n");
    printf("Roll No: %d (Total Marks: %d)\n", highest_total_roll, highest_total);

    return 0;
}
