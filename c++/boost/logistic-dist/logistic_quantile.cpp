#include <cmath>
#include <iostream>
#include <boost/math/distributions/logistic.hpp>

using namespace std;
using boost::math::logistic_distribution;

using namespace boost::math::policies;

typedef policy<promote_double<false>> no_double_promotion;
typedef policy<promote_double<true>> double_promotion;


template <typename T>
T logit_scipy(T p) {
    // The standard formula is log(p/(1 - p)), but this expression
    // loses precision near p=0.5, as does log(p) - log1p(-p).
    // We use the standard formula away from p=0.5, and use
    // log1p(2*(p - 0.5)) - log1p(-2*(p - 0.5)) around p=0.5, which
    // provides very good precision in this interval.
    if (p < 0.3 || p > 0.65) {
        return std::log(p / (1 - p));
    } else {
        T s = 2 * (p - 0.5);
        return std::log1p(s) - std::log1p(-s);
    }
};

template <typename T>
T logit_proposed(T p) {
    // The standard formula is log(p/(1 - p)), but this expression
    // loses precision near p=0.5, as does log(p) - log1p(-p).
    // A mathematically equivalent expression is 2*atanh(2*p - 1),
    // which maintains high precision for p > 0.25.  For small p,
    // precision is lost in the expression 2*p - 1.  So for p < 0.25,
    // we use the standard formula.
    if (p < 0.25) {
        return std::log(p / (1 - p));
    } else {
        return 2*std::atanh(2*p - 1);
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
        double w = logit_proposed(p);
        cout << fixed << setw(25) << setprecision(17) << p << " ";
        cout << scientific << setw(25) << setprecision(17) << q << " ";
        cout << scientific << setw(25) << setprecision(17) << r << " ";
        cout << scientific << setw(25) << setprecision(17) << s << " ";
        cout << scientific << setw(25) << setprecision(17) << w << " ";
        cout << endl;
    }

    return 0;
}
