
#include <iostream>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/erf.hpp>

using namespace std;


using boost::math::erf_inv;


int main()
{
    double pvals[] = {-3.5, -1.0, -0.25, 0, 0.96, 1.0, 2.3};

    cout << "  p         erf_inv(p)" << endl;
    for (const auto &p : pvals) {
        cout << scientific << setprecision(1) << setw(9) << p;
        try {
            double x = erf_inv(p);
            cout << "   " << setprecision(17) << setw(24) << x << endl;
        } catch (const exception& e) {
            cout << "   ***" << endl;
            cerr << "*** Caught: " << e.what() << endl;
            cerr << "*** Type: " << typeid(e).name() << endl;
        }
    }
}
