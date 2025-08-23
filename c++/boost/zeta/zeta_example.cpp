#include <iostream>
#include <boost/math/special_functions/zeta.hpp>


int main()
{
    double x = 15.0;
    double z = boost::math::zeta(x);
    std::cout << std::setprecision(17) << z << std::endl;
}