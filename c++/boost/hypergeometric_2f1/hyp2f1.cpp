#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

using namespace boost::math;

double hyp2f1(double a1, double a2, double b1, double x)
{
    double h = hypergeometric_pFq({a1, a2}, {b1}, x);
    return h;
}


int main(int argc, char *argv[])
{
    if (argc != 5) {
        std::cout << "use: " << argv[0] << " a1 a2 b x\n";
        return -1;
    }

    double a1 = std::stod(argv[1]);
    double a2 = std::stod(argv[2]);
    double b = std::stod(argv[3]);
    double x = std::stod(argv[4]);

    double y = hyp2f1(a1, a2, b, x);

    std::cout << std::scientific << std::setw(24)
        << std::setprecision(17) << y << std::endl;

    return 0;
}
