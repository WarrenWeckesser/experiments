#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>

using namespace std;
using boost::math::non_central_t;


int main()
{
    tuple<double, double> params[] = {
        //   df,    nc
        {   400,  1.25},
        {   250,  25.0},
        {  1000,   2.0},
        {100000,   2.0}
    };

    cout << "     df       nc         mean     variance" << endl;
    for (auto [df, nc] : params) {
        auto ncf = non_central_t(df, nc);
        double m = mean(ncf);
        double v = variance(ncf);

        cout << fixed << setw(10) << setprecision(1) << df << " ";
        cout << fixed << setw(10) << setprecision(4) << nc << " ";
        cout << fixed << setw(21) << setprecision(15) << m << " ";
        cout << fixed << setw(21) << setprecision(15) << v << endl;
    }

    return 0;
}
