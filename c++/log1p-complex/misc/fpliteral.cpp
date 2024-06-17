
#include <cstdio>
#include "../print_floating_point.h"

int main()
{
    double w = 0.3;
    printf("w = ");
    print_value(w);
    printf("\n");
    double x = 0.3L;
    printf("x = ");
    print_value(x);
    printf("\n");
    long double y = 0.3;
    printf("y = ");
    print_value(y);
    printf("\n");
    long double z = 0.3L;
    printf("z = ");
    print_value(z);
    printf("\n");
}