#include <iostream>
#include <iomanip>
#include <boost/math/distributions/negative_binomial.hpp>

using namespace std;
using namespace boost::math;
using boost::math::negative_binomial_distribution;

int main(int argc, char *argv[])
{
    double x = 2321.0;
    double r = 269.0;
    double p = 0.0405;
    negative_binomial_distribution<> dist(r, p);
    double c = cdf(dist, x);
    cout << c << endl;
    return 0;
}
