
#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;


int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "use: " << argv[0] << " a b p\n";
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double p = std::stod(argv[3]);
    if (p <= 0 || p >= 1) {
        std::cerr << "must have 0 < p < 1" << std::endl;
        return -1;
    }

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
