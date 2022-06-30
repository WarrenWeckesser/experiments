
#include <boost/math/distributions/skew_normal.hpp>
using boost::math::skew_normal_distribution;

#include <iostream>
using namespace std;


int main()
{
    double loc = 0.0;
    double scale =  1.0;
    double shape[] = {2.0, 2.0, 5.0};
    double x[] = {1.0, -4.0, -2.0};

    int n = sizeof(x) / sizeof(x[0]);

    cout << "        x           shape   pdf                     cdf" << endl;
    cout << scientific;
    for (int i = 0; i < n; ++i) {
        double p = pdf(skew_normal_distribution(loc, scale, shape[i]), x[i]);
        double c = cdf(skew_normal_distribution(loc, scale, shape[i]), x[i]);
        cout << fixed << setw(12) << setprecision(4) << x[i] << " ";
        cout << fixed << setw(12) << setprecision(4) << shape[i] << " ";
        cout << scientific << setprecision(17) << p << " ";
        cout << scientific << setprecision(17) << c << endl;
    }
}
