#include <iostream>
#include <ginac/ginac.h>

using namespace std;
using namespace GiNaC;

int main()
{
    symbol x("x"), y("y");
    ex poly;

    for (int i=0; i<3; ++i)
        poly += factorial(i+16)*pow(x,i)*pow(y,2-i);
    poly *= Pi;

    cout << poly << endl;

    bool uses_pow = poly.has(pow(wild(1), wild(2)));
    bool uses_atan2 = poly.has(atan2(wild(1), wild(2)));
    bool uses_pi = poly.has(Pi);
    cout << "poly uses pow:   " << uses_pow << endl;
    cout << "poly uses atan2: " << uses_atan2 << endl;
    cout << "poly uses Pi:    " << uses_pi << endl;

    return 0;
}
