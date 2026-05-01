#include <stdio.h>

int main(void) {
	float x, y, z;
	float sum, avg, max, min;

	printf("Enter three values: ");
	scanf("%f %f %f", &x, &y, &z);

	sum = x + y + z;
	avg = sum / 3.0f;

	max = x;
	if (y > max) max = y;
	if (z > max) max = z;

	min = x;
	if (y < min) min = y;
	if (z < min) min = z;

	printf("Sum: %.3f\n", sum);
	printf("Average: %.3f\n", avg);
	printf("Largest: %.3f\n", max);
	printf("Smallest: %.3f\n", min);
}
