#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/bessel.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    double nu = 2;

    std::cout << "Zeros of J_nu(x)" << std::endl;
    std::cout << "  nu k x" << std::endl;
    for (int m = 1; m <= 6; ++m) {
       double z = cyl_bessel_j_zero(nu, m);
       std::cout << std::setw(4) << std::fixed << std::setprecision(1) << nu << " " << m << " "
                 << std::scientific << std::setw(16)
                 << std::setprecision(12) << z << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Zeros of Y_nu(x)" << std::endl;
    std::cout << "  nu k x" << std::endl;
    for (int m = 1; m <= 6; ++m) {
       double z = cyl_neumann_zero(nu, m);
       std::cout << std::setw(4) << std::fixed << std::setprecision(1) << nu << " " << m << " "
                 << std::scientific << std::setw(16)
                 << std::setprecision(12) << z << std::endl;
    }
    return 0;
}
