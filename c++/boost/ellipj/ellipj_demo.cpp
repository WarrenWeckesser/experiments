#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/jacobi_elliptic.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    double m = 0.99999999997;
    double u = 50.0;
    double sn, cn, dn;

    sn = jacobi_elliptic(m, u, &cn, &dn);
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << sn << std::endl;
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << cn << std::endl;
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << dn << std::endl;
    return 0;
}
