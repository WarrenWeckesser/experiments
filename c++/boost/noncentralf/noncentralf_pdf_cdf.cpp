
// Compile with
//   g++ -std=c++1z noncentralf.cpp -o noncentralf

#include <string>
#include <iostream>
#include <boost/math/distributions/non_central_f.hpp>

using namespace std;
using boost::math::non_central_f;


int main(int argc, char *argv[])
{
    if (argc != 5 ) {
        printf("use: %s x dfn dfd nc\n", argv[0]);
        return -1;
    }

    double x = std::stod(argv[1]);
    double dfn = std::stod(argv[2]);
    double dfd = std::stod(argv[3]);
    double nc = std::stod(argv[4]);

    auto ncf = non_central_f(dfn, dfd, nc);
    double p = pdf(ncf, x);
    double c = cdf(ncf, x);

    cout << "pdf: " << setw(24) << setprecision(16) << p << endl;
    cout << "cdf: " << setw(24) << setprecision(16) << c << endl;

    return 0;
}
