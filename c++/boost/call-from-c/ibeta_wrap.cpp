
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;

extern "C" {

float boost_float_ibeta(float a, float b, float x) {
	return ibeta(a, b, x);
}

double boost_double_ibeta(double a, double b, double x) {
	return ibeta(a, b, x);
}

long double boost_longdouble_ibeta(long double a, long double b, long double x) {
	return ibeta(a, b, x);
}
}