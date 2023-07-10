#include <iostream>
#include <cln/real.h>
#include <cln/real_io.h>

using namespace cln;

//
// Cumulative distribution function of the Benktander type II probability
// distribution.  See, for example,
//     https://en.wikipedia.org/wiki/Benktander_type_II_distribution
//
const cl_F benktander2_cdf(const cl_F& x, const cl_F& a, const cl_F& b)
{
    return 1 - exp((b - 1)*ln(x) + (a/b)*(1 - exp(b*ln(x))));
}

int main(int argc, char *argv[])
{
    cl_F x = "1.0000001_80";
    cl_F a = "0.5_80";
    cl_F b = "0.9999_80";
    cl_F y = benktander2_cdf(x, a, b);

    std::cout << "x = " << x << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "\ncdf(x, a, b) = " << y << std::endl;
}