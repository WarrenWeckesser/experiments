#include <cmath>
#include <iostream>
#include <boost/math/distributions/logistic.hpp>

using namespace std;
using boost::math::logistic_distribution;

using namespace boost::math::policies;

typedef policy<promote_double<false>> no_double_promotion;
typedef policy<promote_double<true>> double_promotion;



template <typename T>
T logit_scipy(T x) {
    // The standard formula is log(x/(1 - x)), but this expression
    // loses precision near x=0.5, as does log(x) - log1p(-x).
    // We use the standard formula away from p=0.5, and use
    // log1p(2*(x - 0.5)) - log1p(-2*(x - 0.5)) around p=0.5, which
    // provides very good precision in this interval.
    if (x < 0.3 || x > 0.65) {
        return std::log(x / (1 - x));
    } else {
        T s = 2 * (x - 0.5);
        return std::log1p(s) - std::log1p(-s);
    }
};


int main()
{
    double location = 0.0;
    double scale = 1.0;
    logistic_distribution<double, no_double_promotion> dist(location, scale);
    double delta = 0.00025;
    for (double p = delta; p < 1.0; p += delta) {
        double q = quantile(dist, p);
        double r = 2*std::atanh(2*p - 1);
        double s = logit_scipy(p);
        cout << fixed << setw(25) << setprecision(17) << p << " ";
        cout << scientific << setw(25) << setprecision(17) << q << " ";
        cout << scientific << setw(25) << setprecision(17) << r << " ";
        cout << scientific << setw(25) << setprecision(17) << s << " ";
        cout << endl;
    }

    return 0;
}
