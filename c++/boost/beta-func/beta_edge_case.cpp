
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

using namespace std;


int main(int argc, char *argv[])
{
    double a = 9.0;
    double b = 1e-204;
    double x = 0.925;
    double y = boost::math::beta(a, b, x);
    std::cout << std::scientific << std::setw(24)
              << std::setprecision(17) << y << std::endl;

    return 0;
}
