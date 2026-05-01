#include <stdio.h>

int main(void){
	
	int num;
	int a, b, c, d;

	printf("Enter a four digits: ");
	scanf("%d", &num);

	a = num % 10;
	b = (num / 10) % 10;
	c = (num / 100) % 10;
	d = num / 1000;

	printf("%d\n", a+b+c+d);
}
