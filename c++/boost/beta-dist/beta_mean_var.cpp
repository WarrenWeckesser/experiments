
// Compile with
//   g++ -std=c++1z beta_mean_var.cpp -o beta_mean_var

#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;


int main()
{
    tuple<double, double> params[] = {
        //    a,     b
        {   0.50,  1.25 },
        {   3.00,  5.00 },
        {  99.00,  0.25 },
        {    1.0,  1.00 }
    };

    cout << "         a          b                  mean              variance" << endl;
    for (auto [a, b] : params) {
        beta_distribution<> dist(a, b);
        double m = mean(dist);
        double v = variance(dist);

        cout << fixed << setw(10) << setprecision(1) << a << " ";
        cout << fixed << setw(10) << setprecision(4) << b << " ";
        cout << fixed << setw(21) << setprecision(15) << m << " ";
        cout << fixed << setw(21) << setprecision(15) << v << endl;
    }

    return 0;
}
