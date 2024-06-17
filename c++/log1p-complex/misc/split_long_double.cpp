#include <cstdio>
#include <cmath>
#include <limits>

#include "../print_floating_point.h"


int main()
{
    if (std::numeric_limits<long double>::digits != 106) {
        printf("This program requires a platform where the 'long double'\n");
        printf("format is IBM double-double.\n");
        exit(-1);
    }
    // long double x = -0.52941176470588235294117647058823L;
    long double x = 0.6L;

    printf("x:          ");
    print_value(x);
    printf("\n");

    double xhi = *((double *) &x);
    double xlo = *(((double *) (&x))+1);
    printf("xhi:        ");
    print_value(xhi);
    printf("\n");
    printf("xlo:        ");
    print_value(xlo);
    printf("\n");

    long double x2 = static_cast<long double>(xhi) + static_cast<long double>(xlo);
    printf("xhi + xlo:  ");
    print_value(x2);
    printf("\n");

    return 0;
}
