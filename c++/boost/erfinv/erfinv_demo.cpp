
#include <iostream>
#include <boost/math/special_functions/erf.hpp>

using boost::math::erf_inv;
using namespace std;


int main()
{
    double p = 1e-20;
    double x = erf_inv(p);
    cout << scientific << setprecision(15) << x << endl;
    // 8.86226925452758027e-21
}
