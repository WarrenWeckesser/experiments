
// Compile with
//   g++ -std=c++20 genextreme_mean_var.cpp -o genextreme_mean_var
// (Add -I <boost directory> if it is not in a standard location.)

#include <iostream>
#include <boost/math/distributions/generalized_extreme_value.hpp>

using boost::math::mean;
using boost::math::variance;
using boost::math::generalized_extreme_value_distribution;

using Real = double;

int main()
{
    std::tuple<Real, Real, Real> params[] = {
        //  mu,  sigma,    xi
        {  0.0,    1.0,  1e-3 },
        {  0.0,    1.0,  1e-6 },
        {  0.0,    1.0,  1e-9 },
        {  0.0,    1.0, 1e-12 }
    };

    std::cout << "        mu      sigma      xi            mean                       variance" << std::endl;
    for (auto [mu, sigma,
         xi] : params) {
        generalized_extreme_value_distribution<> dist(mu, sigma, xi);
        double m = mean(dist);
        double v = variance(dist);

        std::cout << std::fixed << std::setw(10) << std::setprecision(1) << mu << " ";
        std::cout << std::fixed << std::setw(10) << std::setprecision(4) << sigma << "  ";
        std::cout << std::scientific << std::setw(10) << std::setprecision(6) << xi << " ";
        std::cout << std::fixed << std::setw(26) << std::setprecision(21) << m << " ";
        std::cout << std::fixed << std::setw(26) << std::setprecision(21) << v << std::endl;
    }

    return 0;
}
