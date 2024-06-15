#include <cln/cln.h>
#include <iostream>


cln::cl_F longdouble_to_cl_f(long double x, int prec)
{
    cln::float_format_t precision = cln::float_format(prec);
    double x0 = x;
    double x1 = (double)(x - x0);
    cln::cl_F y0 = cln::cl_float(x0, precision);
    cln::cl_F y1 = cln::cl_float(x1, precision);
    return y0 + y1;
}

int main(int argc, char *argv[])
{
    std::cout << "Pure CLN calculation of log1p(z)" << std::endl;

    cln::float_format_t precision = cln::float_format(60);
    // cln::cl_F one = cln::cl_float(1.0, precision);
    // cln::cl_F x = cln::cl_float(-0.375, precision);
    // cln::cl_F y = cln::cl_float(0.7806247497992445, precision);
    cln::cl_F one = longdouble_to_cl_f(1.0L, 60);
    cln::cl_F x = longdouble_to_cl_f(-0.2L, 60);
    cln::cl_F y = longdouble_to_cl_f(0.6L, 60);
    cln::cl_N zp1 = cln::complex(x + one, y);
    //std::cout << "z:   " << z << std::endl;
    //auto zp1 = z + one;
    std::cout << "zp1: " << zp1 << std::endl;
    auto w = cln::log(zp1);
    std::cout << "w:   " << w << std::endl;

    // long double t = 0.333333333333333333333L;
    // cln::cl_F s = longdouble_to_cl_f(t, 60);
    // std::cout << "s: " << s << std::endl;

    return 0;
}
