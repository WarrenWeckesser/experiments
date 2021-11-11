
// Compile with
//   g++ -std=c++1z beta_pdf.cpp -o beta_pdf

#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;


int main()
{
    double a = 11.0;
    double b = 99990.0;
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    beta_distribution<> dist(a, b);
    for (double x = 0.00045; x <= 0.00055; x += 0.000001) {
        double p = cdf(dist, x);
        double q = cdf(complement(dist, x));
        cout << fixed << setw(11) << setprecision(8) << x << " ";
        cout << scientific << setw(23) << setprecision(16) << p << " ";
        cout << scientific << setw(23) << setprecision(16) << q << " ";
        //if (p == 0) {
        //    cout << "!!!";
        //}
        cout << endl;
    }

    return 0;
}
