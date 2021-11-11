#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

int main(int argc, char *argv[])
{
	double a, b, x, y;

	a = 0.5;
	b = 5e-7;
	x = 0.9999990000010001;
	y = ibeta(a, b, x);
	std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
	return 0;
}
