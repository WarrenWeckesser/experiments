#include <iostream>
#include <array>
#include <boost/math/distributions/cauchy.hpp>
using boost::math::cauchy_distribution;
using boost::math::complement;

using namespace std;


int main()
{
    double loc = 0.0;
    double scale =  1.0;
    const double x[] = {-1e17, -5e14, 0, 5e14, 1e17};

    cout << "        x                     cdf                       sf" << endl;
    cout << scientific;
    for (auto& x1: x) {
        double c = cdf(cauchy_distribution(loc, scale), x1);
        double s = cdf(complement(cauchy_distribution(loc, scale), x1));
        cout << setw(16) << setprecision(8) << x1 << " ";
        cout << setw(25) << setprecision(17) << c << " ";
        cout << setw(25) << setprecision(17) << s << endl;
    }
}
