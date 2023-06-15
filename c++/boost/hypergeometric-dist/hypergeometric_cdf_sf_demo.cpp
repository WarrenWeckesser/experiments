#include <cstdio>
#include <boost/math/distributions/hypergeometric.hpp>

using namespace std;
using namespace boost::math;
using boost::math::hypergeometric_distribution;

int main(int argc, char *argv[])
{
    // unsigned long total = 10000000000000000;
    // unsigned long total = 1874919424;
    unsigned long total = 3567587328;
    unsigned long ngood = 100000;
    unsigned long nsample = 10;
    hypergeometric_distribution<> dist(ngood, nsample, total);
    unsigned k = 9;
    double c = cdf(dist, k);
    printf("c = %24.17e\n", c);
    double s = cdf(complement(dist, k));
    printf("s = %24.17e\n", s);
    return 0;
}
