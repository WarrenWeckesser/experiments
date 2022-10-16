#include <string>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

using namespace boost::math;


int main(int argc, char *argv[])
{
    if (argc != 4 ) {
        printf("use: hyp1f1 a b x\n");
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double x = std::stod(argv[3]);

    double y1 = hypergeometric_1F1(a, b, x);
    std::cout << std::scientific << std::setw(25)
        << std::setprecision(17) << y1 << std::endl;

    double y2 = hypergeometric_pFq({a}, {b}, x);
    std::cout << std::scientific << std::setw(25)
        << std::setprecision(17) << y2 << std::endl;

    return 0;
}
