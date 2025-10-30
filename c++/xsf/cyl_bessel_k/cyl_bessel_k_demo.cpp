// Requires at least C++17.

#include <complex>
#include <iomanip>
#include <iostream>
#include <xsf/bessel.h>

int main(int argc, char *argv[])
{
    using namespace std;

    double nu = 0.0;
    double re = 0.0;
    double im = 0.0;

    if (argc > 1) {
        nu = stod(argv[1]);
        if (argc > 2) {
            re = stod(argv[2]);
            if (argc > 3) {
                im = stod(argv[3]);
            }
        }
    }
    complex<double> z(re, im);
    complex<double> w = xsf::cyl_bessel_k(nu, z);

    cout << setprecision(15);
    cout << w.real() << " " << w.imag() << endl;
    return 0;
}
