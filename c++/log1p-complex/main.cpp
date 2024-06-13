#include <cstdio>
#include <cmath>
#include <complex>

#include "log1p_complex.h"
#include "print_stuff.h"


int main()
{
    // complex<long double> z(-0.2L, 0.6L);
    complex<double> z(-0.2, 0.6);
    // complex<float> z(-0.2f, 0.6f);
    // complex<long double> z(-1.0L, 1.0L);
    // complex<long double> z(-8.0L/13.0L, 12.0L/13.0L);
    // complex<long double> z(-9.0L/17.0L, 15.0L/17.0L);
    // complex<long double> z(-9999.0L/10000.0L, 1.0L);
    
    auto w = log1p_complex(z);
    printf("w: ");
    print_value(w.real());
    printf(" + ");
    print_value(w.imag());
    printf("i\n");

    return 0;
}
