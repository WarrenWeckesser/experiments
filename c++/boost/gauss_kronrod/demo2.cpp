#include <cstdio>
#include <string>
#include <boost/math/quadrature/gauss_kronrod.hpp>

using namespace boost::math::quadrature;



int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("use: %s a b\n", argv[0]);
        exit(-1);
    }
    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);

    printf("a = %25.17e\n", a);
    printf("b = %25.17e\n", b);

    double error;
    auto integrand = [](double t) {return 1.0/(t*t*t);};
    double q = gauss_kronrod<double, 31>::integrate(integrand, a, b, 20, 5e-14, &error);

    printf("gauss_kronrod: %25.17e    error: %25.17e\n", q, error);
    printf("exact value:   %25.17e\n", (1/(a*a) - 1/(b*b))/2);

    return 0;
}
