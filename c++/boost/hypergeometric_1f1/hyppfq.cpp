#include <string>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

using namespace boost::math;


int main(int argc, char *argv[])
{
    if (argc != 4 ) {
        printf("use: hyppfq a b x\n");
        return -1;
    }

    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);
    double x = std::stod(argv[3]);

    double y = hypergeometric_pFq({a}, {b}, x);
    std::cout << "pFq = " << std::scientific << std::setw(25)
        << std::setprecision(17) << y << std::endl;

    return 0;
}
