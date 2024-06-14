#include <cstdio>
#include <limits>

#include "../print_stuff.h"

int main()
{
    long double eps = std::numeric_limits<long double>::epsilon();
    long double one_over_eps = 1.0L/eps;
    printf("eps   = ");
    print_value(eps);
    printf("\n");
    printf("1/eps = ");
    print_value(one_over_eps);
    printf("\n");
}

