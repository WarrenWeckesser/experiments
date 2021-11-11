
// Compile with
//   g++ -std=c++1z beta_pdf.cpp -o beta_pdf

#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;


int main()
{
    double a = 2.0;
    double b = 1.5;
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    beta_distribution<> dist(a, b);
    for (double x = 0; x <= 1.0; x += 0.125) {
        double p = pdf(dist, x);
        cout << fixed << setw(8) << setprecision(3) << x << " ";
        cout << scientific << setw(23) << setprecision(16) << p << " ";
        if (p == 0) {
            cout << "!!!";
        }
        cout << endl;
    }

    return 0;
}
