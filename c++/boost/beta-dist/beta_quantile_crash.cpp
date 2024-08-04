#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;

int main(int argc, char *argv[])
{
    cout << BOOST_VERSION << endl;

    double a = 5.0;
    double b = 5.0;
    double p = 0.5;
 
    beta_distribution<> dist(a, b);
    double x = quantile(dist, p);

    cout << scientific << setw(23) << setprecision(16) << x << endl;

    return 0;
}
