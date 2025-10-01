//
//  float_as_integer_ratio.cpp
//
//  Large integers are handled with the cl_I class from the CLN library.
//
//  Compile with:
//     g++ -std=c++17 float_as_integer_ratio.cpp -o float_as_integer_ratio
//         $(pkg-config --cflags cln) $(pkg-config --libs cln)
//

#include <cln/cln.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

#include <cstring>
#include <cstdio>
#include <charconv>
#include <system_error>

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
// Convert a nonnegative C++ floating point whose value is known to be an
// integer into a cl_I.
//
// CLN might have a way to do this directly that I missed in the docs.
// This function must work with `float`, `double` and `long double`.
//
template <typename T>
cln::cl_I
float_to_cl_I(T x)
{
    cln::cl_I result = 0;
    cln::cl_I k = 0;
    while (x > 0) {
        T rem = std::fmod(x, static_cast<T>(2));
        if (rem > 0) {
            result += cln::cl_I(1) << k;
        }
        x = std::floor(x / 2);
        k += 1;
    }
    return result;
}

//
// This functon was copied from the Python source code, edited to use
// CLN for large integers, and updated to work with `float`, `double` or
// `long double`.
// The function is not thoroughly tested!
//
// Special cases:
//   inf -> 1/0
//  -inf -> -1/0
//   nan -> 0/0
//
template <typename T>
std::pair<cln::cl_I, cln::cl_I>
float_as_integer_ratio(T x)
{
    cln::cl_I numerator;
    cln::cl_I denominator;

    if (std::isinf(x)) {
        if (x < 0) {
            numerator = -1;
        }
        else {
            numerator = 1;
        }
        denominator = 0;
    }
    else if (std::isnan(x)) {
        numerator = 0;
        denominator = 0;
    }
    else if (x == -1 || x == 0 || x == 1) {
        // Probably don't need to handle this as a special case.
        numerator = static_cast<int>(x);
        denominator = 1;
    }
    else {
        int sgn = x < 0 ? -1 : 1;  // x == 0 was already handled.
        x *= sgn;
        int exponent;
        T float_part = std::frexp(x, &exponent);
        cln::cl_I expon = exponent;
        for (int i = 0; i < 400 && float_part != std::floor(float_part); ++i) {
            float_part *= 2;
            expon--;
        }
        numerator = float_to_cl_I(float_part);
        denominator = 1;
        if (expon > 0) {
            numerator *= 1 << expon;
        }
        else {
            denominator *= 1 << -expon;
        }
        numerator *= cln::cl_I(sgn);
    }
    return std::pair(numerator, denominator);
}

template <typename T>
void
show_ratio(T x)
{
    auto ratio = float_as_integer_ratio(x);
    std::cout << "ratio is " << ratio.first << " / " << ratio.second << std::endl;
}

int
main(int argc, char **argv)
{
    float x = 1310.23;
    double y = 0.1;
    long double z = 1.0L / 3.0L;

    print_value(x);
    std::cout << std::endl;
    show_ratio(x);

    std::cout << std::endl;
    print_value(y);
    std::cout << std::endl;
    show_ratio(y);

    std::cout << std::endl;
    print_value(z);
    std::cout << std::endl;
    show_ratio(z);

    return 0;
}
