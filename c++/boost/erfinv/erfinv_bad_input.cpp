
#include <iostream>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/erf.hpp>

using namespace std;


using boost::math::erf_inv;


int main()
{
    double pvals[] = {-1.0, 1.0};

    cout << "  p       erf_inv(p)" << endl;
    for (const auto &p : pvals) {
        double x = -999.25;
        try {
            x = erf_inv(p);
        } catch (const exception& e) {
            cerr << "*** Caught: " << e.what() << endl;
            cerr << "*** Type: " << typeid(e).name() << endl;
        }
        cout << scientific << setprecision(1) << p << " " << setprecision(17) << x << endl;
    }
}
