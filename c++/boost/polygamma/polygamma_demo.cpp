#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/polygamma.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "use: " << argv[0] << " n x" << std::endl;
        exit(0);
    }
    int n = std::stoi(argv[1]);
    double x = strtold(argv[2], nullptr);

    double y = polygamma(n, x);
    std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
    return 0;
}
