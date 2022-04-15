
#include <iostream>
#include <boost/math/special_functions/powm1.hpp>

using namespace std;


using boost::math::powm1;


int main()
{
    double xvals[] = {-1.2, 0.0};
    double yvals[] = {-2.0, -1.5, 0.0, 0.5, 1.0, 2.0};

    for (const auto &x : xvals) {
        for (const auto &y : yvals) {
            cout << scientific << setprecision(4) << setw(11) << x << "  " << y;
            try {
                double p = powm1(x, y);
                cout << "   " << setprecision(17) << setw(24) << p << endl;
            } catch (const exception& e) {
                cout << "   ***" << endl;
                cerr << "*** Caught: " << e.what() << endl;
                cerr << "*** Type: " << typeid(e).name() << endl;
            }
        }
    }
}
