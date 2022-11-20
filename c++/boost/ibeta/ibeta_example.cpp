#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "use: " << argv[0] << " a b x\n";
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double x = std::stod(argv[3]);

    double y1 = ibeta(a, b, x);
    std::cout << std::scientific << std::setw(24)
        << std::setprecision(17) << y1 << std::endl;

    double y1full = y1 * beta(a, b);
    std::cout << std::scientific << std::setw(24)
        << std::setprecision(17) << y1full << std::endl;

    if (a > 0 && b > 0) {
        double y2 = beta(a, b, x);
        std::cout << std::scientific << std::setw(24)
            << std::setprecision(17) << y2 << std::endl;
    }

    return 0;
}
