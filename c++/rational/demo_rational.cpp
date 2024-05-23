#include <iostream>
#include <cstdint>

#include "rational.hpp"


int main()
{
    double x = 1.0;
    auto y1 = rational::fixed_rational(x);
    std::cout  << y1 << std::endl;
}
