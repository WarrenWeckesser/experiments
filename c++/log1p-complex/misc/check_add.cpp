#include <cstdio>
#include <limits>

#include "../print_floating_point.h"

int main()
{
    long double xhi = -0.5294117647058824;
    long double xlo = 6.530723674265628e-18;
    long double x = xhi + xlo;
    printf("x   = ");
    print_value(x);
    printf("\n");
}

