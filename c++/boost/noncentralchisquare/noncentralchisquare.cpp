
#include <boost/math/distributions/non_central_chi_squared.hpp>
using boost::math::non_central_chi_squared;

#include <iostream>
using std::cout; using std::endl; using std::setw;
using std::scientific;
using std::fixed;
using std::setprecision;

int main()
{
    double cs[] =  {3,  25, 250, 0.001,  100,  150,  160};
    double nu[] =  {3,   8,  25,     8, 13.5, 13.5, 13.5};
    double lam[] = {1, 250, 200,    40,   10,   10,   10};

    int n = sizeof(cs) / sizeof(cs[0]);

    cout << scientific;
    for (int i = 0; i < n; ++i) {
        double c = cdf(non_central_chi_squared(nu[i], lam[i]), cs[i]);

        cout << fixed << setw(8) << setprecision(4) << cs[i] << " ";
        cout << fixed << setw(8) << setprecision(4) << nu[i] << " ";
        cout << fixed << setw(8) << setprecision(4) << lam[i] << " ";
        cout << scientific << setprecision(17) << c << endl;
    }
}
