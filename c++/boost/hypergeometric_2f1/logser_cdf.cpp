#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

double hyp2f1(double a1, double a2, double b1, double x)
{
    double h = hypergeometric_pFq({a1, a2}, {b1}, x);
    return h;
}


// Technically, k should be an integer...

double logser_cdf_hyp2f1(double k, double p)
{
    double r = pow(p, k+1) * hyp2f1(1, k+1, k+2, p) / (k+1);
    return 1.0 + r / log1p(-p);
}


// double logser_cdf_beta(double k, double p)
// {
//     return 1.0 + beta(k+1, 0, p) / log1p(-p);
// }

// double logser_cdf_beta2(double k, double p)
// {
//     double b = (beta(k+1, 1,  p) - p*beta(k, 1, p))/(1 - p);
//     return 1.0 + b / log1p(-p);
// }

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "use: " << argv[0] << " k p\n";
        return -1;
    }

    double k = std::stod(argv[1]);
    double p = std::stod(argv[2]);
    if (p <= 0 || p >= 1) {
        std::cerr << "must have 0 < p  < 1" << std::endl;
        return -1;
    }

    // double c1 = logser_cdf_beta2(k, p);
    // std::cout << "beta2:  " << std::scientific << std::setw(22)
    //     << std::setprecision(17) << c1 << std::endl;

    double c2 = logser_cdf_hyp2f1(k, p);
    std::cout << "cdf using hyp2f1: " << std::scientific << std::setw(22)
        << std::setprecision(17) << c2 << std::endl;
    return 0;
}
