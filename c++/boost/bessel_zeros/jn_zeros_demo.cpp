#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/bessel.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    double nu = 281;

    for (int m = 1; m <= 6; ++m) {
       double z = cyl_bessel_j_zero(nu, m);
       std::cout << std::scientific << std::setw(16)
                 << std::setprecision(12) << z << std::endl;
    }
    return 0;
}
