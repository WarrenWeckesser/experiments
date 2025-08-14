#include <iostream>
#include <boost/math/distributions/logistic.hpp>

using namespace std;
using namespace boost::math::policies;
using boost::math::logistic_distribution;

typedef policy<
    promote_float<false>,
    promote_double<false>
> no_promotion;

typedef policy<
    promote_float<true>,
    promote_double<true>
> with_promotion;

int main()
{
    logistic_distribution<double, no_promotion> dist;
    logistic_distribution<double, with_promotion> dist_promote;
    logistic_distribution<long double, no_promotion> dist_longdouble;
    double p = 2049.0/4096;
    double ref = 0.0009765625776102258;  // Verified with both Wolfram Alpha and mpmath
    double x = quantile(dist, p);
    double xpromote = quantile(dist_promote,  p);
    double xld = static_cast<double>(quantile(dist_longdouble, static_cast<long double>(p)));
    cout << fixed << setw(25) << setprecision(17) << p << " p" << endl;
    cout << scientific << setw(25) << setprecision(17) << x << " double"<< endl;
    cout << scientific << setw(25) << setprecision(17) << xpromote << " double with promotion" << endl;
    cout << scientific << setw(25) << setprecision(17) << xld << " long double"<< endl;
    cout << scientific << setw(25) << setprecision(17) << ref << " reference"<< endl;

    return 0;
}
