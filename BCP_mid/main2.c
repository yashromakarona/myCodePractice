#include <stdio.h>

int main(void){
	int n;
	float sum;

	printf("Enter a value of n: ");
	scanf("%d", &n);
	
	sum = 0;
	for (int i = 1; i < n+1; i++){
		sum += 1.0f / i;
	}
	
	printf("%.3f\n", sum);
	
}
