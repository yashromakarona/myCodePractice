#include <stdio.h>

void read_data(float *a, float *b, char *op) {
    printf("Enter calculation (e.g., 5 + 3): ");
    scanf("%f %c %f", a, op, b);
}
float add(float a, float b) { return a + b; }
float sub(float a, float b) { return a - b; }
float mul(float a, float b) { return a * b; }
float div(float a, float b) { return b != 0 ? a / b : 0; }

float calculate(float a, float b, char op) {
    switch(op) {
        case '+': return add(a, b);
        case '-': return sub(a, b);
        case '*': return mul(a, b);
        case '/': return div(a, b);
        default: return 0;
    }
}
void display(float res) { printf("Result: %.2f\n", res); }

void run_calculator() {
    float n1, n2; char op;
    read_data(&n1, &n2, &op);
    display(calculate(n1, n2, op));
}

int main() {
    run_calculator();
    return 0;
}
