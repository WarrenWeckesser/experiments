#include <iostream>
#include <complex>
#include <cln/cln.h>

#include <cstring>
#include <cstdio>
#include <charconv>
//#include <cstring>
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

using namespace cln;
using namespace std;


class doubledouble_t {
public:
    double upper;
    double lower;
};

doubledouble_t clf_to_doubledouble_approx(const cl_F& x)
{
    double x0 = double_approx(x);
    double x1 = double_approx(x - x0);
    return doubledouble_t{x0, x1};
}

//
// Emulate the creation of a cl_F from a `long double` on a platform
// where `long double` is implemented using IBM double-double format.
//
cl_F str_to_clf_via_doubledouble(const char *text, const float_format_t fmt)
{
    cl_read_flags flags;
    flags.syntax = syntax_float;
    flags.float_flags.default_float_format = fmt;
    cl_F t = read_float(flags, text, NULL, NULL);
    doubledouble_t tdd = clf_to_doubledouble_approx(t);
    return cl_float(tdd.upper, fmt) + cl_float(tdd.lower, fmt);
}


int main(int argc, char *argv[])
{
    int dps = 200;
    float_format_t fmt = float_format(dps);

    const char xtext[] = "-0.2";
    const char ytext[] = "0.6";

    auto x = str_to_clf_via_doubledouble(xtext, fmt);
    auto y = str_to_clf_via_doubledouble(ytext, fmt);
    auto z = cln::complex(x, y);

    auto w = log(z + 1);

    // Downcast the real and imaginary parts of w so we can print
    // the result with a reasonably sized text output.
    float_format_t outfmt = float_format(33);
    auto w_re = cl_float(realpart(w), outfmt);
    auto w_im = cl_float(imagpart(w), outfmt);
    cout << "On a platform where 'long double' is IBM double-double, with" << endl;
    cout << "z = complex(" << xtext << "L, " << ytext << "L), ";
    cout << "w = log1p(z) should produce" << endl;
    cout << "w.real:   " << w_re << " [...]" << endl;
    cout << "w.imag:   " << w_im << " [...]" << endl;

    return 0;
}
