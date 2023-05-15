#include <string>
#include <iostream>
#include <iomanip>

#define BOOST_MATH_MAX_ROOT_ITERATION_POLICY  400
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY      false
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;


int main(int argc, char *argv[])
{
    if (argc != 4 ) {
        printf("use: %s a b p\n", argv[0]);
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double p = std::stod(argv[3]);

    double x = ibeta_inv(a, b, p);
    std::cout << std::scientific << std::setw(20)
        << std::setprecision(10) << a;
    std::cout << std::scientific << std::setw(20)
        << std::setprecision(10) << b;
    std::cout << std::scientific << std::setw(20)
        << std::setprecision(10) << p;
    std::cout << std::scientific << std::setw(26)
        << std::setprecision(17) << x << std::endl;

    return 0;
}
