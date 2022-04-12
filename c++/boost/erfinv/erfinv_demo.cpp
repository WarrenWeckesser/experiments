
#include <iostream>
#include <boost/math/special_functions/erf.hpp>

using boost::math::erf_inv;
using namespace std;


int main()
{
    double pvals[] = {5e-30, 1e-20, 5e-16, 2e-8, 1e-7, 5e-6};

    cout << "  p       erf_inv(p)" << endl;
    for (const auto &p : pvals) {
        double x = erf_inv(p);
        cout << scientific << setprecision(1) << p << " " << setprecision(17) << x << endl;
    }
}
