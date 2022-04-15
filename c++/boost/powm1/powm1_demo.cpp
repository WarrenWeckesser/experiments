
#include <iostream>
#include <boost/math/special_functions/powm1.hpp>

using boost::math::powm1;
using namespace std;


int main()
{
    double xvals[] = {0.2, 1.05, 19.3};
    double yvals[] = {1e-8, 1.25, 3.5};

    cout << "   x           y            powm1(x, y)" << endl;
    for (const auto &x : xvals) {
        for (const auto &y : yvals) {
            double p = powm1(x, y);
            cout << scientific << setprecision(4) << setw(11) << x << "  " << y
                << setprecision(17) << setw(26) << p << endl;
        }
    }
}
