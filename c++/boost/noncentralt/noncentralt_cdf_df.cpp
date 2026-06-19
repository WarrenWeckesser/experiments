#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>

using namespace std;
using namespace boost::math::policies;
using boost::math::non_central_t_distribution;

typedef policy<promote_double<false>> no_double_promotion;
typedef policy<promote_double<true>> double_promotion;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        cout << "use: " << argv[0] << " nc x\n";
        return -1;
    }

    double nc = stod(argv[1]);
    double x = stod(argv[2]);

    double df = 1e-8;
    for (int k = 1; k < 8000; ++k) {
        auto nct1 = non_central_t_distribution<double, no_double_promotion>(df, nc);
        double p1 = cdf(nct1, x);
        cout << scientific << setw(23) << setprecision(16) << df << " " << p1 << endl;
        df *= 1.00225;
    }
    for (int k = 1; k < 4000; ++k) {
        auto nct1 = non_central_t_distribution<double, no_double_promotion>(df, nc);
        double p1 = cdf(nct1, x);
        cout << scientific << setw(23) << setprecision(16) << df << " " << p1 << endl;
        df += 0.5;
    }
    return 0;
}
