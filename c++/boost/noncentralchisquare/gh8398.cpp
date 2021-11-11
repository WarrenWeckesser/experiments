
#include <boost/math/distributions/non_central_chi_squared.hpp>
using boost::math::non_central_chi_squared;

#include <iostream>
using namespace std;

int main()
{
    double x = 30.0;
    double nu = 20.0;
    double lam = 10.0;

    double c = cdf(non_central_chi_squared(nu, lam), x);
    cout << scientific << setprecision(17) << c << endl;
}
