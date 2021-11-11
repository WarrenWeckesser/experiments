
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
    double p = 1e-11;
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    beta_distribution<> dist(a, b);
    double x = quantile(dist, p);
    double y = quantile(complement(dist, p));
    //cout << fixed << setw(11) << setprecision(8) << x << " ";
    cout << scientific << setw(23) << setprecision(16) << x << " ";
    cout << scientific << setw(23) << setprecision(16) << y << " ";
    cout << endl;

    return 0;
}
