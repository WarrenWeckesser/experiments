#include <string>
#include <iostream>
#include <iomanip>
#include <cfenv>

#define BOOST_MATH_MAX_ROOT_ITERATION_POLICY  400
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY      false
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

void show_fp_exception_flags()
{
    if (std::fetestexcept(FE_DIVBYZERO)) {
        std::cout << " FE_DIVBYZERO";
    }
    // FE_INEXACT is common and not interesting.
    // if (std::fetestexcept(FE_INEXACT)) {
    //     cout << " FE_INEXACT";
    // }
    if (std::fetestexcept(FE_INVALID)) {
        std::cout << " FE_INVALID";
    }
    if (std::fetestexcept(FE_OVERFLOW)) {
        std::cout << " FE_OVERFLOW";
    }
    if (std::fetestexcept(FE_UNDERFLOW)) {
        std::cout << " FE_UNDERFLOW";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    double a = 2e-308;
    double b = 5.0;
    double p = 0.9;

    std::feclearexcept(FE_ALL_EXCEPT);
    double x = ibeta_inv(a, b, p);
    show_fp_exception_flags();

    std::cout << "BOOST_MATH_PROMOTE_DOUBLE_POLICY "
        << BOOST_MATH_PROMOTE_DOUBLE_POLICY << std::endl;
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
