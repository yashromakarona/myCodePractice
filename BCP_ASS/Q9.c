#include <stdio.h>
struct time_struct {
    int hour;
    int minute;
    int second;
};
int main() {
    struct time_struct t;
    t.hour = 16;
    t.minute = 40;
    t.second = 51;
    printf("%02d:%02d:%02d\n", t.hour, t.minute, t.second);
    return 0;
}
