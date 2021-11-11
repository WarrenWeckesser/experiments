#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
	double a, b, x, y, z;

	a = 0;
	b = 3;
	x = 0.5;
	y = ibeta(a, b, x);
	std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
	return 0;
}
