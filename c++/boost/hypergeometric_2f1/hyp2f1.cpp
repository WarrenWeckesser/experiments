#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

using namespace boost::math;

double hyp2f1(double a1, double a2, double b1, double x)
{
    double h = hypergeometric_pFq({a1, a2}, {b1}, x);
    return h;
}


int main(int argc, char *argv[])
{
    double a1, a2, b1, x, y;


	a1 = 0.5;
    a2 = 1.5;
    b1 = 4.0;
    x = 0.5;
    y = hyp2f1(a1, a2, b1, x);
	std::cout << std::scientific << std::setw(16)
        << std::setprecision(12) << y << std::endl;
	return 0;
}
