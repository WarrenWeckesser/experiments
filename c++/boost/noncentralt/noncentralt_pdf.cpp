
// Compile with
//   g++ -std=c++1z noncentralt_pdf.cpp -o noncentralt_pdf

#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>

using namespace std;
using boost::math::non_central_t;


int main(int argc, char *argv[])
{
    if (argc != 4) {
        cout << "use: " << argv[0] << " x df nc\n";
        return -1;
    }

    double x = stod(argv[1]);
    double df = stod(argv[2]);
    double nc = stod(argv[3]);

    cout << "x  = " << x << endl;
    cout << "df = " << df << endl;
    cout << "nc = " << nc << endl;
    auto nct = non_central_t(df, nc);
    double p = pdf(nct, x);
    cout << "p  = " << scientific << setw(23) << setprecision(16) << p << endl;

    return 0;
}
