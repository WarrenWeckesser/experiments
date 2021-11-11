
// Compile with
//   g++ -std=c++1z noncentralt_pdf.cpp -o noncentralt_pdf

#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>

using namespace std;
using boost::math::non_central_t;


int main()
{
    double df = 8.0;
    double nc = 8.5;
    cout << "df = " << df << endl;
    cout << "nc = " << nc << endl;
    auto ncf = non_central_t(df, nc);
    for (double x = -10.0; x < 5.0; x += 1.0) {
        double p = pdf(ncf, x);
        cout << fixed << setw(8) << setprecision(2) << x << " ";
        cout << scientific << setw(23) << setprecision(16) << p << " ";
        if (p == 0) {
            cout << "!!!";
        }
        cout << endl;
    }

    return 0;
}
