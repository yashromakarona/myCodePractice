#include <stdio.h>
int fib(int n) {
    if (n == 1 || n == 2) return 1;
    return fib(n - 1) + fib(n - 2);
}
void print_fib(int n) {
    printf("First %d Fibonacci numbers: ", n);
    for (int i = 1; i <= n; i++) printf("%d ", fib(i));
    printf("\n");
}
int main() {
    print_fib(5);
    print_fib(10);
    print_fib(15);
    return 0;
}
