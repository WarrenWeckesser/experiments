
#include <boost/math/distributions/poisson.hpp>
using boost::math::poisson_distribution;

#include <iostream>
using namespace std;


int main()
{
    double mean = 2719.3;

    cout << "    k      cdf" << endl;
    for (int i = 2716; i < 2722; ++i) {
        double c = cdf(poisson_distribution(mean), i);
        cout << setw(5) << i << " ";
        cout << c << endl;
    }
}
