#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/polygamma.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    unsigned n = 25;
    double x = 355.0/113.0;
    double y = polygamma(n, x);
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
    return 0;
}
