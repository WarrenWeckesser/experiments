#include <cstdio>
#include <cmath>
#include <complex>

#include "log1p_complex.h"
#include "print_floating_point.h"

using namespace std;

int main()
{
    // complex<long double> z(-0.2L, 0.6L);
    // complex<double> z(-0.2, 0.6);
    // complex<float> z(-0.2f, 0.6f);
    // complex<long double> z(-1.0L, 1.0L);
    // complex<long double> z(-8.0L/13.0L, 12.0L/13.0L);
    // complex<long double> z(-9.0L/17.0L, 15.0L/17.0L);
    // complex<long double> z(-9999.0L/10000.0L, 1.0L);

    // NOTE! The constants here are intentionally not given
    // the L suffix, so they are *double* (not *long double*)
    // values that are assigned to the long double variables.
    // The test cases created this way are for platforms
    // where long double is implemented using IBM double-double.
    //long double xhi = -0.5294117647058824;
    //long double xlo = 6.530723674265628e-18;
    //long double yhi = 0.8823529411764706;
    //long double ylo = 2.6122894697062506e-17;
    //
    //long double xhi = -1.2500111249898924;
    //long double xlo = 7.43809241772238e-17;
    //long double yhi = 0.6666666666666666;
    //long double ylo = 3.700743415417188e-17;
    //complex<long double> z(xhi + xlo, yhi + ylo);

    long double xhi = -0.2;
    //long double xlo = 0.0;
    long double xlo = 1.1102230246251566e-17;
    long double yhi = 0.6;
    //long double ylo = 0.0;
    long double ylo = 2.22044604925031326e-17;
    complex<long double> z(xhi + xlo, yhi + ylo);

    // long double xhi = -0.5;
    // long double xlo = 3.3e-18;
    // long double yhi = 0.25;
    // long double ylo = 4.9e-18;
    // complex<long double> z(xhi + xlo, yhi + ylo);
    
    printf("z: ");
    print_value(z.real());
    printf(" + ");
    print_value(z.imag());
    printf("i\n");

    auto w = log1p_complex::log1p_complex(z);

    printf("w: ");
    print_value(w.real());
    printf(" + ");
    print_value(w.imag());
    printf("i\n");

    return 0;
}
