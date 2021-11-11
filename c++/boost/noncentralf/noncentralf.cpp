
// Compile with
//   g++ -std=c++1z noncentralf.cpp -o noncentralf

#include <iostream>
#include <boost/math/distributions/non_central_f.hpp>

using namespace std;
using boost::math::non_central_f;


int main()
{
    tuple<double, double, double> params[] = {
        // dfn, dfd,   nc
        {    2,   6,    4},
        {   10,  12,  2.4},
        {  250,  25, 27.5}
    };

    cout << "     dfn      dfd       nc         mean     variance" << endl;
    for (auto [dfn, dfd, nc] : params) {
        auto ncf = non_central_f(dfn, dfd, nc);
        double m = mean(ncf);
        double v = variance(ncf);

        cout << fixed << setw(8) << setprecision(4) << dfn << " ";
        cout << fixed << setw(8) << setprecision(4) << dfd << " ";
        cout << fixed << setw(8) << setprecision(4) << nc << " ";
        cout << fixed << setw(12) << setprecision(7) << m << " ";
        cout << fixed << setw(12) << setprecision(7) << v << endl;
    }

    return 0;
}
