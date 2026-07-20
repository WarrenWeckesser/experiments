#include <string>
#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/expint.hpp>

using namespace boost::math;


int main(int argc, char *argv[])
{
    if (argc != 2 ) {
        printf("use: expint x\n");
        return -1;
    }

    double x = std::stod(argv[1]);

    double y = expint(x);
    std::cout << "expint = " << std::scientific << std::setw(25)
        << std::setprecision(17) << y << std::endl;

    return 0;
}
