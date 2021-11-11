
#include <boost/math/distributions/non_central_chi_squared.hpp>
using boost::math::non_central_chi_squared;

#include <iostream>
using std::cout; using std::endl; using std::setw;
using std::scientific;
using std::fixed;
using std::setprecision;

int main()
{
    double cs[] =  {100, 200, 1000};
    double nu[] =  {1, 1, 1};
    double lam[] = {1000, 2000, 10000};

    int n = sizeof(cs) / sizeof(cs[0]);

    cout << scientific;
    for (int i = 0; i < n; ++i) {
        double c = cdf(non_central_chi_squared(nu[i], lam[i]), cs[i]);

        cout << fixed << setw(12) << setprecision(4) << cs[i] << " ";
        cout << fixed << setw(12) << setprecision(4) << nu[i] << " ";
        cout << fixed << setw(12) << setprecision(4) << lam[i] << " ";
        cout << scientific << setprecision(17) << c << endl;
    }
}
