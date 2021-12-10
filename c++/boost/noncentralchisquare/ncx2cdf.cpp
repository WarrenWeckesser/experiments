
#include <boost/math/distributions/non_central_chi_squared.hpp>
using boost::math::non_central_chi_squared;

#include <iostream>
using std::cout; using std::endl; using std::setw;
using std::scientific;
using std::fixed;
using std::setprecision;

int main()
{
    double x[] =  {10, 100, 200, 1000};
    double nu[] =  {1, 1, 1, 1};
    double lam[] = {15, 1000, 2000, 10000};

    int n = sizeof(x) / sizeof(x[0]);

    cout << "        x            nu           lam    cdf" << endl;
    cout << scientific;
    for (int i = 0; i < n; ++i) {
        double c = cdf(non_central_chi_squared(nu[i], lam[i]), x[i]);

        cout << fixed << setw(12) << setprecision(4) << x[i] << " ";
        cout << fixed << setw(12) << setprecision(4) << nu[i] << " ";
        cout << fixed << setw(12) << setprecision(4) << lam[i] << " ";
        cout << scientific << setprecision(17) << c << endl;
    }
}
