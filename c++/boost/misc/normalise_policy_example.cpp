#include <iostream>
#include <iomanip>
#include "boost/math/special_functions/lambert_w.hpp"

using namespace boost::math;
using namespace boost::math::policies;

typedef policy<promote_float<false>> policy_no_promotion;
typedef policy<promote_float<true>> policy_promotion;

int main(int argc, char *argv[])
{
    float w0;
    float x = -0.3678;
    w0 = lambert_w0(x, policy_no_promotion());
    std::cout << std::setw(16) << std::setprecision(10) << w0 << "  policy_no_promotion\n";
    w0 = lambert_w0(x, policy_promotion());
    std::cout << std::setw(16) << std::setprecision(10) << w0 << "  policy_promotion\n";
    w0 = lambert_w0(x, normalise<policy_promotion, promote_float<false>, promote_double<false>>::type());
    std::cout << std::setw(16) << std::setprecision(10) << w0 << "  override promote_float in policy_promotion\n";
    return 0;
}
