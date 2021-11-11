#include <iostream>
#include <iomanip>
#include <boost/math/distributions/negative_binomial.hpp>

using namespace std;
using namespace boost::math;
using boost::math::negative_binomial_distribution;

int main(int argc, char *argv[])
{
    double r = 10.0;
    double p = 0.9;
    negative_binomial_distribution<> dist(r, p);
    double q = quantile(dist, 0.51);
    cout << q << endl;
    return 0;
}
