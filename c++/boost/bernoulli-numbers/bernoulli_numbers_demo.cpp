#include <iostream>
#include <boost/math/special_functions/bernoulli.hpp>

using namespace boost::math::policies;

typedef policy<promote_double<false>> no_promote_double;


int main()
{
    // These reference values are from the wikipedia page https://en.wikipedia.org/wiki/Bernoulli_number
    double bnumbers2n[] = {1.0,  1.0/6, -1.0/30, 1.0/42, -1.0/30, 5.0/66, -691.0/2730, 7.0/6, -3617.0/510, 43867.0/798, -174611.0/330};
    int k = 0;
    std::cout << "    n            bernoulli_b2n(n)                   reference" << std::endl;
    std::cout << "-----   -------------------------   -------------------------" << std::endl;
    for (const auto &ref : bnumbers2n) {
        double b = boost::math::bernoulli_b2n<double>(k, no_promote_double());
        std::cout << std::setw(5) << 2*k
                  << std::setw(28) << std::setprecision(17) << b
                  << std::setw(28) << std::setprecision(17) << ref
                  << std::endl;
        k += 1;
    }
    int kvals[] = {50, 75, 100, 125};
    // These references values were computed with mpmath.bernoulli().
    double refvals[] = {-2.8382249570693707e+78, 2.142610125066529e+143, -3.647077264519136e+215, 1.843526146783894e+293};
    for (int i = 0; i < 4; ++i) {
        double b = boost::math::bernoulli_b2n<double>(kvals[i], no_promote_double());
        std::cout << std::setw(5) << 2*kvals[i]
                  << std::setw(28) << std::setprecision(17) << b
                  << std::setw(28) << std::setprecision(17) << refvals[i]
                  << std::endl;
    }
}
