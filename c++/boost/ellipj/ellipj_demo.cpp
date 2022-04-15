#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/jacobi_elliptic.hpp>

using namespace boost::math;
using namespace std;

int main(int argc, char *argv[])
{
    double k = 0.99999999997;
    double u = 50.0;
    double sn, cn, dn;

    sn = jacobi_elliptic(k, u, &cn, &dn);
    cout << setprecision(17);
    cout << "k = " << k << endl;
    cout << "m = k**2 = " << k*k << endl;
    cout << "u = " << u << endl;
    cout << "sn = " << scientific << setw(17) << sn << endl;
    cout << "cn = " << scientific << setw(17) << cn << endl;
    cout << "dn = " << scientific << setw(17) << dn << endl;
    return 0;
}
