#include <math.h>
#include <stdio.h>

#define PI 3.1416
#define MAX 180

int main() {
    int angle;
    float x, y;
    angle = 0;

    printf(" Angle  Cos(angle)\n");
    while (angle <= MAX) {
        x = (PI/MAX) * angle;
        y = cos(x);
        printf("%5d %11.4f\n", angle, y);
        angle = angle + 10;

    }
    return 0;
}
