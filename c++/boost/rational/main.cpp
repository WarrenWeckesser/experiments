#include <iostream>
#include "boost/math/tools/rational.hpp"

int main(int argc, char *argv[])
{
    const double a[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    const double b[] = {1.0, 1.0, 1.0, 1.0, 1.0};

    double x1 = 1e80;
    double y1 = boost::math::tools::evaluate_rational(a, b, x1);

    double x2 = -1e80;
    double y2 = boost::math::tools::evaluate_rational(a, b, x2);
    
    std::cout << y1 << " " << y2 << std::endl;

    return 0;
}