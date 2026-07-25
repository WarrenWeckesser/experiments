#include <iostream>
#include <boost/math/special_functions/bernoulli.hpp>

using namespace boost::math::policies;

typedef policy<promote_double<false>> no_promote_double;


int main()
{
    double bnumbers2n[] = {1.0,  1.0/6, -1.0/30, 1.0/42, -1.0/30, 5.0/66, -691.0/2730, 7.0/6, -3617.0/510, 43867.0/798, -174611.0/330};
    int n = 0;
    std::cout << "    n        bernoulli_b2n(n)               reference" << std::endl;
    std::cout << "-----   ---------------------   ---------------------" << std::endl;
    for (const auto &ref : bnumbers2n) {
        double b = boost::math::bernoulli_b2n<double>(n, no_promote_double());
        std::cout << std::setw(5) << 2*n
                  << std::setw(24) << std::setprecision(17) << b
                  << std::setw(24) << std::setprecision(17) << ref
                  << std::endl;
        n += 1;
    }
}
