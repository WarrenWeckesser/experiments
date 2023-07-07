
#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;
using namespace boost::math::policies;

// Define a policy that sets ::errno on overflow, and does
// not promote double to long double internally:
typedef policy<promote_double<false>> my_policy;

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "use: " << argv[0] << " a b x\n"
                  << "If x is the character s, the value x=1e-310 is used.\n";
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double x;
    if (argv[3][0] == 's') {
        x = 1e-310;
    }
    else {
        x = std::stod(argv[3]);
    }
    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    cout << "x = " << x << endl;

    beta_distribution<double, my_policy> dist(a, b);
    double p = pdf(dist, x);
    cout << "p = " << p << endl;

    return 0;
}

