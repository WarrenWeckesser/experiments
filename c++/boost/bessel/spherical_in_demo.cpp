//
// spherical_in_demo.cpp
//
// Requires C++20, but only for std::numbers::pi.  Replace that constant
// to use an older standard.
//

#include <cstdio>
#include <charconv>
#include <cmath>
#include <numbers>
#include <string>
#include <system_error>

#include <boost/math/special_functions/bessel.hpp>

using namespace boost::math;  // for cyl_bessel_i

template<typename T>
static inline void
print_value(const T x)
{
    char buf[100];
    const std::to_chars_result res = std::to_chars(buf, buf + sizeof(buf), x);
    if (res.ec == std::errc{}) {
        printf("%.*s", static_cast<int>(res.ptr - buf), buf);
    }
    else {
        printf("<to_chars() failed!>");
    }
}

//
// Modified spherical Bessel function of the first kind, i_nu(x).
//
double spherical_in(double nu, double x)
{
    return cyl_bessel_i(nu + 0.5, x) * std::sqrt(std::numbers::pi / (2 * x));
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        printf("use: %s nu x1 x2 n\n", argv[0]);
        return -1;
    }

    double nu = std::stod(argv[1]);
    double x1 = std::stod(argv[2]);
    double x2 = std::stod(argv[3]);
    int n = std::stoi(argv[4]);

    for (int k = 0; k < n; ++k) {
        double x = x1 + k / static_cast<double>(n - 1) * (x2 - x1);
        double y = spherical_in(nu, x);
        print_value(nu);
        printf(" ");
        print_value(x);
        printf(" ");
        print_value(y);
        printf("\n");
    }
    return 0;
}
