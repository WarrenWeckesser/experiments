#include <iostream>
#include <boost/math/distributions/beta.hpp>

using boost::math::beta_distribution;

int main()
{
    float a = 1.5f;
    float b = 3.0f;
    beta_distribution<float> dist(a, b);
    auto m = mean(dist);

    std::cout << "mean is " << m << std::endl;

    return 0;
}
