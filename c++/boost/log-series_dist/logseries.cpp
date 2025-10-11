
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

// TODO: Check numerical behavior of these functions over the
// full ranges of the inputs, especially near p = 1.

double logseries_cdf(int64_t k, double p)
{
    using boost::math::beta;

    return 1.0 + beta(k + 1, 1e-300, p) / std::log1p(-p);
}

double logseries_sf(int64_t k, double p)
{
    using boost::math::beta;
    
    return -beta(k + 1, 1e-300, p) / std::log1p(-p);
}

int main(int argc, char *argv[])
{
    double p = 1e-8;
    int64_t k = 3;
    double sf = logseries_sf(k, p);
    std::cout << std::setprecision(17);
    std::cout << "sf = " << sf << std::endl;

    return 0;
}