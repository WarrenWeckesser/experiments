
// Compile with
//   g++ -std=c++1z logistic_cdf.cpp -o logistic_cdf

#include <iostream>
#include <boost/math/distributions/logistic.hpp>

using namespace std;
using boost::math::logistic_distribution;


int main()
{
    double location = 0.0;
    double scale = 1.0;
    cout << "location = " << location << endl;
    cout << "scale    = " << scale << endl;
    logistic_distribution<> dist(location, scale);
    for (double x = -750.0; x <= 750.0; x += 50.0) {
        double p = cdf(dist, x);
        double q = cdf(complement(dist, x));
        cout << fixed << setw(15) << setprecision(8) << x << " ";
        cout << scientific << setw(25) << setprecision(16) << p << " ";
        cout << scientific << setw(25) << setprecision(16) << q << " ";
        cout << endl;
    }

    return 0;
}
