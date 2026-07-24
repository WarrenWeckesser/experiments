#include <string>
#include <iostream>
#include <cmath>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/constants/constants.hpp>

//
// Warning: this function loses precision for moderately large values of eta.
// At eta = 20 the result has just a couple digits of precision, and for not
// much larger etc, all precision is lost.
//
double gompertz_noncentral_moment2(double eta)
{
    using namespace std;
    double s = log(eta) + boost::math::constants::euler<double>();
    double f = s*s + boost::math::constants::pi_sqr_div_six<double>()
                - 2 * eta * boost::math::hypergeometric_pFq({1, 1, 1}, {2, 2, 2}, -eta);
    return exp(eta) * f;
}


int
main(int argc, char *argv[])
{
    if (argc == 1) {
        std::cout << "use: " << argv[0] << " x1 x2 x3 ..." << std::endl;
        return -1;
    }
    for (int k = 1; k < argc; ++k) {
        double eta = std::stod(argv[k]);
        double m2 = gompertz_noncentral_moment2(eta);

        std::cout << "eta = " << std::setw(15) << std::setprecision(15) << eta;
        std::cout << "    m2 = " << std::setprecision(17) << m2 << std::endl;
    }
    return 0;
}
