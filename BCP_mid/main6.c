#include <stdio.h>
#include <math.h>

int main() {

    float price_shirts = 40.50, price_banana = 0.80, price_orange_juice = 2.30;
    int qty_shirts = 0, qty_banana = 0, qty_orange_juice = 0;
    int choice, qty;
    float total_bill = 0.0;

    printf("--- Supermarket Product List ---\n");
    printf("1. Shirts ($40.50)\n");
    printf("2. Banana ($0.80)\n");
    printf("3. Orange Juice ($2.30)\n");


    while (1) {
        printf("\nEnter the product number to purchase (1-3) (0 to finish): ");
        scanf("%d", &choice);

        if (choice == 0) {
            break;
        }

        switch (choice) {
            case 1:
                printf("Input the quantity of Shirts: ");
                scanf("%d", &qty);
                qty_shirts += qty;
                break;
            case 2:
                printf("Input the quantity of Banana: ");
                scanf("%d", &qty);
                qty_banana += qty;
                break;
            case 3:
                printf("Input the quantity of Orange Juice: ");
                scanf("%d", &qty);
                qty_orange_juice += qty;
                break;
            default:
                printf("Invalid product number.\n");
                break;
        }
    }


    total_bill = (price_shirts * qty_shirts) + (price_banana * qty_banana) + (price_orange_juice * qty_orange_juice);
    float rounded_total = ceil(total_bill);

    printf("\n--- Detailed Receipt ---\n");
    if (qty_shirts > 0) printf("Product: Shirts | Price: $40.50 | Quantity: %d\n", qty_shirts);
    if (qty_banana > 0) printf("Product: Banana | Price: $0.80 | Quantity: %d\n", qty_banana);
    if (qty_orange_juice > 0) printf("Product: Orange Juice | Price: $2.30 | Quantity: %d\n", qty_orange_juice);

    printf("------------------------------------------------\n");
    printf("Total (before round up): $%.2f\n", total_bill);
    printf("Final Payment (rounded up): $%.0f\n", rounded_total);

    return 0;
}
