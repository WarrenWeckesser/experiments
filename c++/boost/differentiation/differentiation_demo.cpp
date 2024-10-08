#include <math.h>
#include <iostream>
#include <limits>
#include <boost/math/differentiation/finite_difference.hpp>


double func(double x)
{
    return exp(-10*x)*sin(x) + 0.25;
}


double func_deriv(double x)
{
    return exp(-10*x)*(cos(x) - 10*sin(x));
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "use: " << argv[0] << " x0 x1 x2 ...\n";
        std::cout << "At least one value must be provided." << std::endl;
        return -1;
    }

    std::vector<double> v;

    for (int k = 1; k < argc; ++k) {
        double arg;
        try {
            arg = std::stod(argv[k]);
        }
        catch(...) {
            std::cout << "Error: unable to convert '" << argv[k]
                      << "' to double." << std::endl;
            return -1;
        }
        v.push_back(arg);
    }

    std::cout << std::setw(20) << "x";
    std::cout << std::setw(20) << "f'(x) (exact)";
    std::cout << std::setw(20) << "f'(x) (estimate)";
    std::cout << std::setw(20) << "rel. error";
    std::cout << std::setw(20) << "error estimate";
    std::cout << std::endl;
    for (const auto& x: v) {
        double relerr;
        double exact = func_deriv(x);
        double error_est;
        double deriv_est = boost::math::differentiation::finite_difference_derivative(func, x, &error_est);
        if (exact != 0) {
            relerr = fabs(exact - deriv_est)/fabs(exact);
        }
        else {
            relerr = std::numeric_limits<double>::quiet_NaN();
        }
        std::cout << std::setw(20) << std::setprecision(12) << x;
        std::cout << std::setw(20) << std::setprecision(12) << exact;
        std::cout << std::setw(20) << std::setprecision(12) << deriv_est;
        std::cout << std::setw(20) << std::setprecision(12) << relerr;
        std::cout << std::setw(20) << std::setprecision(12) << error_est << std::endl;
    }

    return 0;
}
