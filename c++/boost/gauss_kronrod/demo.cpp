#include <cstdio>
#include <string>
#include <boost/math/quadrature/gauss_kronrod.hpp>

using namespace boost::math::quadrature;


//
// Compute the PDF of the truncated normal distribution
// using numerical integration.  The formula is
//
//     pdf(x) = 1/{integral wrt. t of exp((x**x-t*t)/2) from t=a to t=b}
//
double truncnorm_pdf(double x, double a, double b)
{
    double error;
    auto integrand = [x](double t) {return std::exp((x*x - t*t)/2);};

    double q = gauss_kronrod<double, 15>::integrate(integrand, a, b, 8, 3e-13, &error);
    printf("q = %25.17e\n", q);
    printf("error = %25.17e\n", error);
    return 1.0 / q;
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf("use: %s x a b\n", argv[0]);
        exit(-1);
    }
    double x = std::stod(argv[1]);
    double a = std::stod(argv[2]);
    double b = std::stod(argv[3]);
    double pdf = truncnorm_pdf(x, a, b);

    printf("x = %25.17e\n", x);
    printf("a = %25.17e\n", a);
    printf("b = %25.17e\n", b);

    printf("pdf = %25.17e\n", pdf);

    return 0;
}
