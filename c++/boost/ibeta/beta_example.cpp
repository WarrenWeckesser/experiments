#define BOOST_MATH_OVERFLOW_ERROR_POLICY errno_on_error

#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;
using namespace boost::math::policies;

// typedef policy<overflow_error<errno_on_error>> my_policy;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "use: " << argv[0] << " a b\n";
        return -1;
    }

    double a;
    if (argv[1][0] == 's') {
        a = 1e-310;
    }
    else {
        a = std::stod(argv[1]);
    }
    double b = std::stod(argv[2]);

    double y = beta(a, b);

    std::cout << "errno = " << errno << std::endl;

    std::cout << std::scientific << std::setw(24)
        << std::setprecision(17) << y << std::endl;

    return 0;
}
