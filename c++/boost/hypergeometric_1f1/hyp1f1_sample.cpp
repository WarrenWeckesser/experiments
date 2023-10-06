#include <string>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

using namespace boost::math;


int main(int argc, char *argv[])
{
    if (argc != 6) {
        std::cerr << "use: hyp1f1 a b0 b1 n x\n" << std::endl;
        return -1;
    }

    double a = std::stod(argv[1]);
    double b0 = std::stod(argv[2]);
    double b1 = std::stod(argv[3]);
    int n = std::stoi(argv[4]);
    double x = std::stod(argv[5]);

    if (n < 2) {
        std::cerr << "n must be greater than 1" << std::endl;
        return -2;
    }

    std::cout << std::scientific << std::setw(25) << std::setprecision(17);
    for (int k = 0; k < n; ++k) {
        double b = b0 + k*(b1 - b0)/(n - 1);
        double y1 = hypergeometric_1F1(a, b, x);
        double y2 = hypergeometric_pFq({a}, {b}, x);

        std::cout << a << " ";
        std::cout << b << " ";
        std::cout << x << " ";
        std::cout << y1 << " ";
        std::cout << y2;
        std::cout << std::endl;
    }

    return 0;
}
