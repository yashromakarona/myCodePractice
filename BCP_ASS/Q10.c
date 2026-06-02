#include <stdio.h>
#include <string.h>

struct hotel {
    char name[50];
    char address[100];
    int grade;
    float room_charge;
    int num_rooms;
};

void print_by_grade(struct hotel hotels[], int count, int grade) {
    struct hotel temp;

    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (hotels[i].room_charge > hotels[j].room_charge) {
                temp = hotels[i];
                hotels[i] = hotels[j];
                hotels[j] = temp;
            }
        }
    }

    printf("\n--- Hotels of Grade %d (Sorted by Charge) ---\n", grade);
    for (int i = 0; i < count; i++) {
        if (hotels[i].grade == grade) {
            printf("Name: %s, Address: %s, Charge: %.2f\n",
                   hotels[i].name, hotels[i].address, hotels[i].room_charge);
        }
    }
}

void print_by_charge(struct hotel hotels[], int count, float max_charge) {
    printf("\n--- Hotels with Charge < %.2f ---\n", max_charge);
    for (int i = 0; i < count; i++) {
        if (hotels[i].room_charge < max_charge) {
            printf("Name: %s, Grade: %d, Charge: %.2f\n",
                   hotels[i].name, hotels[i].grade, hotels[i].room_charge);
        }
    }
}

int main() {
    struct hotel hotels[3] = {
        {"Grand Seoul", "Seoul", 5, 250000.0, 200},
        {"Busan Ocean", "Busan", 4, 120000.0, 100},
        {"Jeju Resort", "Jeju", 5, 180000.0, 150}
    };

    print_by_grade(hotels, 3, 5);
    print_by_charge(hotels, 3, 200000.0);

    return 0;
}