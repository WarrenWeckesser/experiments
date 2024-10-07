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
    double x = std::stod(argv[3]);

    double p = ibeta(a, b, x);
    std::cout << "a  = " << std::scientific << std::setw(26)
        << std::setprecision(17) << a << std::endl;
    std::cout << "p  = " << std::scientific << std::setw(26)
        << std::setprecision(17) << p << std::endl;
    double a1 = ibeta_inva(b, x, p);
    std::cout << "a1 = " << std::scientific << std::setw(26)
        << std::setprecision(17) << a1 << std::endl;

    /*
    std::cout << std::scientific << std::setw(20)
        << std::setprecision(10) << b;
    std::cout << std::scientific << std::setw(20)
        << std::setprecision(10) << p;
    std::cout << std::scientific << std::setw(26)
        << std::setprecision(17) << x << std::endl;
    */
    
    return 0;
}
