#include <stdio.h>
#include <string.h>

// 호텔 구조체 정의
struct hotel {
    char name[2];
    char address[3];
    int grade;
    float room_charge;
    int num_rooms;
};

// (a) 특정 등급의 호텔을 요금 오름차순으로 정렬하여 출력하는 함수
void print_by_grade(struct hotel hotels[], int count, int grade) {
    struct hotel temp;

    // 전체 배열을 요금 오름차순으로 정렬 (버블 정렬 사용)
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

// (b) 주어진 금액보다 저렴한 호텔을 출력하는 함수
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
    // 기능 테스트를 위한 임의의 호텔 데이터
    struct hotel hotels[4] = {
        {"Grand Seoul", "Seoul", 5, 250000.0, 200},
        {"Busan Ocean", "Busan", 4, 120000.0, 100},
        {"Jeju Resort", "Jeju", 5, 180000.0, 150}
    };

    print_by_grade(hotels, 3, 5);      // 5성급 호텔 요금순 출력
    print_by_charge(hotels, 3, 200000.0); // 20만 원 미만 호텔 출력

    return 0;
}