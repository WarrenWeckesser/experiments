#include <string>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>

using namespace boost::math;

//
// double hyp2f1(double a1, double a2, double b1, double x)
// {
//     double h = hypergeometric_pFq({a1, a2}, {b1}, x);
//     return h;
// }
//


int main(int argc, char *argv[])
{
    double a, b, x, y;

    if (argc != 4 ) {
        printf("use: hyp1f1 a b x\n");
        return -1;
    }
    a = std::stod(argv[1]);
    b = std::stod(argv[2]);
    x = std::stod(argv[3]);
    y = hypergeometric_1F1(a, b, x);
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
    return 0;
}
