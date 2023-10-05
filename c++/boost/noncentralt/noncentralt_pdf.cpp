
// Compile with
//   g++ -std=c++1z noncentralt_pdf.cpp -o noncentralt_pdf

#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>

using namespace std;
using namespace boost::math::policies;
using boost::math::non_central_t_distribution;

typedef policy<promote_double<false>> no_double_promotion;
typedef policy<promote_double<true>> double_promotion;

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
    auto nct1 = non_central_t_distribution<double, no_double_promotion>(df, nc);
    double p1 = pdf(nct1, x);
    cout << "p  = " << scientific << setw(23) << setprecision(16) << p1
         << " (with no_double_promotion policy)" << endl;

    auto nct2 = non_central_t_distribution<double, double_promotion>(df, nc);
    double p2 = pdf(nct2, x);
    cout << "p  = " << scientific << setw(23) << setprecision(16) << p2
         << " (with double_promotion policy)" << endl;

    return 0;
}
