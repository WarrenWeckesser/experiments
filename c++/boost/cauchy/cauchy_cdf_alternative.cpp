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
    const double x[] = {-1e80, -1e24, -1e17, -5e14, 1000.0, -3.0, -0.25, -1e-8, -3e-18, -1e-23, 0, 1e-23, 3e-18, 1e-8, 0.25, 3.0, 1000.0, 5e14, 1e17};

    cout << "        x                     cdf                       alternative   abs(delta)" << endl;
    cout << scientific;
    for (auto& x1: x) {
        double c = cdf(cauchy_distribution(loc, scale), x1);
        double c2 = std::atan2(1, -x1)/boost::math::constants::pi<double>();
        // double s = cdf(complement(cauchy_distribution(loc, scale), x1));
        cout << setw(16) << setprecision(8) << x1 << " ";
        cout << setw(25) << setprecision(17) << c << " ";
        cout << setw(25) << setprecision(17) << c2 << " ";
        cout << setw(25) << setprecision(12) << std::abs(c - c2) << " ";
        // cout << setw(25) << setprecision(17) << s << endl;
        cout << endl;
    }
}
