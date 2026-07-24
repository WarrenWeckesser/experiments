#include <iostream>
#include <boost/math/special_functions/gamma.hpp>


int main()
{
    double x = 15.0;
    double a = 1e-8;

    std::cout << std::setprecision(17) << x << std::endl;    
    std::cout << std::setprecision(17) << a << std::endl;

    double p = boost::math::gamma_p(a, x);

    std::cout << std::setprecision(17) << p << std::endl;

    double a2 = boost::math::gamma_p_inva(x, p);

    std::cout << std::setprecision(17) << a2 << std::endl;

    return 0;
}
