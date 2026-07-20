#include <string>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/expint.hpp>

using namespace boost::math;


int main(int argc, char *argv[])
{
    if (argc != 3 ) {
        printf("use: expn n x\n");
        return -1;
    }

    double n = std::stod(argv[1]);
    double x = std::stod(argv[2]);

    double y = expint(n, x);
    std::cout << "expn = " << std::scientific << std::setw(25)
        << std::setprecision(17) << y << std::endl;

    return 0;
}
