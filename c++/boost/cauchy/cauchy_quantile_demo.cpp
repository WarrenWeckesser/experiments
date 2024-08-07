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
    const double pnear1[] = {0.999, 0.999999, 0.999999999, 0.999999999999};
    const double pnear0[] = {0.001, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21};

    cout << "        p                     quantile" << endl;
    cout << scientific;
    for (auto& p1: pnear1) {
        double x = quantile(cauchy_distribution(loc, scale), p1);

        cout << setw(23) << setprecision(15) << p1 << " ";
        cout << setw(28) << setprecision(18) << x << endl;
    }

    cout << "            p                 complement quantile" << endl;
    cout << scientific;
    for (auto& p1: pnear0) {
        double x = quantile(complement(cauchy_distribution(loc, scale), p1));

        cout << setw(23) << setprecision(8) << p1 << " ";
        cout << setw(28) << setprecision(18) << x << endl;
    }
}
