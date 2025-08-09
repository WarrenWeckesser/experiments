//
//  float_as_integer_ratio.cpp
//
//  Large integers are handled with the cl_I class from the CLN library.
//
//  Compile with:
//     g++ float_as_integer_ratio.cpp -o float_as_integer_ratio -lcln
//

#include <cmath>
#include <iostream>
#include <utility>
#include <cln/cln.h>

//
// This functon was copied from the Python source code, edited to use
// CLN for large integers, and updated to work with float or double.
// The function is not thoroughly tested!
//
template<typename T>
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
        int exponent;
        T float_part = frexp(x, &exponent);
        for (int i = 0; i < 300 && float_part != floor(float_part); ++i) {
            float_part *= 2;
            exponent--;
        }
        numerator = static_cast<long long>(float_part);
        denominator = 1;

        cln::cl_I p = std::abs(exponent);
        if (exponent > 0) {
            numerator *= 1 << p;
        }
        else {
            denominator *= 1 << p;
        }
    }
    return std::pair(numerator, denominator);
}

template<typename T>
void show_ratio(T x)
{
    auto ratio = float_as_integer_ratio(x);
    std::cout << "ratio is " << ratio.first << " / " << ratio.second << std::endl;
}

int main(int argc, char **argv)
{
    float x = 1310.23;
    double y = -2.3e-6;

    show_ratio(x);
    show_ratio(18334.3f);
    show_ratio(3.4025e38f);
    show_ratio(0.50000006f);
    show_ratio(y);
    show_ratio(0.0);
    show_ratio(0.0f);
    show_ratio(-1.0);
    show_ratio(NAN);
    show_ratio(static_cast<double>(NAN));
    show_ratio(INFINITY);
    show_ratio(-INFINITY);
    return(0);
}
