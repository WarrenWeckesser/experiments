
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>

using namespace boost::math::policies;
using boost::math::tgamma_ratio;
using namespace std;


typedef policy<underflow_error<throw_on_error>> my_policy;

int main()
{
    double xvals[] = {120.0, 150.0, 180.0};
    double yvals[] = {160.0, 180.0, 200.0};

    cout << "     x     y        tgamma_ratio(x, y)" << endl;
    for (const auto &x : xvals) {
        for (const auto &y : yvals) {
            double p = tgamma_ratio(x, y, my_policy());
            cout << setprecision(5) << setw(6) << x << setw(6) << y
                << setprecision(17) << setw(26) << p << endl;
        }
    }
}
