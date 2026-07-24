// From https://github.com/boostorg/math/issues/1429

#include <boost/math/distributions/students_t.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>
#include <cmath>

int main() {
    for (double t : {1e-20, 1e-15, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}) {
        double p = boost::math::cdf(boost::math::students_t_distribution<double>(1.), t);
        double p_simple = 0.5 + std::atan2(t, 1.0) / M_PI;
        std::cout <<  "t: " << std::setprecision(17) << t
                  << ", p: " << std::setprecision(17) << p
                  << ", p simple: " << std::setprecision(17) << p_simple
                  << std::endl;
    }
    return 0;
}
