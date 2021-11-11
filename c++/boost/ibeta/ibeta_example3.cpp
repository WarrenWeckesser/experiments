#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
	double a, b, x, y, z;

	a = 5;
	b = 99996;
	x = 0.999995;
	y = ibeta_inv(a, b, x);
	std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
    // z = ibeta(a, b, y);
    // std::cout << std::scientific << std::setw(16)
    //     << std::setprecision(12) << z << std::endl;
	return 0;
}
