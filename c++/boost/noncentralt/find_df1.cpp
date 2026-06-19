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
    if (argc != 4) {
        cout << "use: " << argv[0] << " p nc x\n";
        return -1;
    }

    double p = stod(argv[1]);
    double nc = stod(argv[2]);
    double x = stod(argv[3]);

    cout << "p  = " << p << endl;
    cout << "nc = " << nc << endl;
    cout << "x  = " << x << endl;

    double df = boost::math::non_central_t_distribution<double, SpecialPolicy>::find_degrees_of_freedom(nc, x, p);
    cout << "df = " << df << endl;
    return 0;
}
