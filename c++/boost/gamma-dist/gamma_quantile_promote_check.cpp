#include <iostream>
#include <boost/math/distributions/gamma.hpp>

using namespace std;
using namespace boost::math::policies;
using boost::math::gamma_distribution;

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
    gamma_distribution<double, no_promotion> dist(1.25);
    gamma_distribution<double, with_promotion> dist_promote(1.25);
    gamma_distribution<long double, no_promotion> dist_longdouble(1.25l);
    double p = 3e-18;
    double ref = 1.0594534539290505e-14;  // Verified with mpmath
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
