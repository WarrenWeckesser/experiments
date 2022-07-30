
// Compile with
//   g++ -std=c++17 beta_mean_var.cpp -o beta_mean_var

#include <iostream>
#include <boost/math/distributions/beta.hpp>

using boost::math::mean;
using boost::math::variance;
using boost::math::beta_distribution;

using Real = long double;

int main()
{
    std::tuple<Real, Real> params[] = {
        //    a,     b
        {   0.50,  1.25 },
        {   3.00,  5.00 },
        {  99.00,  0.25 },
        {   1.00,  1.00 }
    };

    std::cout << "         a          b                  mean              variance" << std::endl;
    for (auto [a, b] : params) {
        beta_distribution<> dist(a, b);
        long double m = mean(dist);
        long double v = variance(dist);

        std::cout << std::fixed << std::setw(10) << std::setprecision(1) << a << " ";
        std::cout << std::fixed << std::setw(10) << std::setprecision(4) << b << " ";
        std::cout << std::fixed << std::setw(26) << std::setprecision(21) << m << " ";
        std::cout << std::fixed << std::setw(26) << std::setprecision(21) << v << std::endl;
    }

    return 0;
}
