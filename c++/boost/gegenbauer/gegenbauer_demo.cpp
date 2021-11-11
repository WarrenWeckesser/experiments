#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/gegenbauer.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    unsigned n = 0;
    double lambda = 0.0;
    double x = 0.5;
    double y = gegenbauer(n, lambda, x);
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
    return 0;
}
