#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>

using namespace std;
using namespace boost::math::policies;
using boost::math::non_central_t_distribution;


typedef boost::math::policies::policy<
    boost::math::policies::promote_float<false >,
    boost::math::policies::promote_double<false >,
    boost::math::policies::max_root_iterations<400 >,
    boost::math::policies::discrete_quantile<boost::math::policies::real > > SpecialPolicy;


int main(int argc, char *argv[])
{
    if (argc != 3) {
        cout << "use: " << argv[0] << " nc x\n";
        return -1;
    }

    double nc = stod(argv[1]);
    double x = stod(argv[2]);

    cout << "nc = " << nc << endl;
    cout << "x  = " << x << endl;

    double deltap = 1.0/16384;
    double p = deltap;
    while (p < 1) {
        double df;
        try {
            df = boost::math::non_central_t_distribution<double, SpecialPolicy>::find_degrees_of_freedom(nc, x, p);
        } catch (...) {
            df = std::numeric_limits<double>::quiet_NaN();
        }
        cout << scientific << setw(23) << setprecision(16) << p << " " << df << endl;
        p += deltap;
    }
    return 0;
}
