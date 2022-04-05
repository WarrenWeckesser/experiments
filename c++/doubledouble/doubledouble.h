//
// A double-double class.
// This is an incomplete translation of the Python code "doubledouble.py"
// by Juraj Sukop, https://github.com/sukop/doubledouble
// That file has the following license and copyright notice:
// # Copyright (c) 2017, Juraj Sukop
// #
// # Permission to use, copy, modify, and/or distribute this software for any
// # purpose with or without fee is hereby granted, provided that the above
// # copyright notice and this permission notice appear in all copies.
// #
// # THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
// # REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
// # AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
// # INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
// # LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// # OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// # PERFORMANCE OF THIS SOFTWARE.
//

#ifndef DOUBLEDOUBLE_H
#define DOUBLEDOUBLE_H

#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstdint>


class DoubleDouble
{
public:

    double upper{0.0};
    double lower{0.0};

    constexpr
    DoubleDouble() {}

    constexpr
    DoubleDouble(double upper, double lower) : upper(upper), lower(lower)
    {}

    constexpr
    DoubleDouble(double upper) : upper(upper)
    {}

    DoubleDouble operator-() const;
    DoubleDouble operator+(const DoubleDouble& x) const;
    DoubleDouble operator-(const DoubleDouble& x) const;
    DoubleDouble operator*(const DoubleDouble& x) const;
    DoubleDouble operator/(const DoubleDouble& x) const;

    DoubleDouble powi(int n) const;
    DoubleDouble exp() const;
    DoubleDouble log() const;
    DoubleDouble sqrt() const;
};


const DoubleDouble dd_e{2.7182818284590452, 1.44564689172925013472e-16};
const DoubleDouble dd_pi{3.1415926535897932, 1.22464679914735317636e-16};


DoubleDouble two_sum_quick(double x, double y)
{
    double r = x + y;
    double e = y - (r - x);
    return DoubleDouble(r, e);
}


DoubleDouble two_sum(double x, double y)
{
    double r = x + y;
    double t = r - x;
    double e = (x - (r - t)) + (y - t);
    return DoubleDouble(r, e);
}


DoubleDouble two_difference(double x, double y)
{
    double r = x - y;
    double t = r - x;
    double e = (x - (r - t)) - (y + t);
    return DoubleDouble(r, e);
}


DoubleDouble two_product(double x, double y)
{
    double r = x*y;
    double e = fma(x, y, -r);
    return DoubleDouble(r, e);
}


DoubleDouble DoubleDouble::operator-() const
{
    return DoubleDouble(-upper, -lower);
}


DoubleDouble DoubleDouble::operator+(const DoubleDouble& x) const
{
    DoubleDouble re = two_sum(upper, x.upper);
    re.lower += lower + x.lower;
    return two_sum_quick(re.upper, re.lower);
}


DoubleDouble DoubleDouble::operator-(const DoubleDouble& x) const
{
    DoubleDouble re = two_difference(upper, x.upper);
    re.lower += lower - x.lower;
    return two_sum_quick(re.upper, re.lower);
}


DoubleDouble DoubleDouble::operator*(const DoubleDouble& x) const
{
    DoubleDouble re = two_product(upper, x.upper);
    re.lower += upper*x.lower + lower*x.upper;
    return two_sum_quick(re.upper, re.lower);
}


DoubleDouble DoubleDouble::operator/(const DoubleDouble& x) const
{
    double r = upper/x.upper;
    DoubleDouble sf = two_product(r, x.upper);
    double e = (upper - sf.upper - sf.lower + lower - r*x.lower)/x.upper;
    return two_sum_quick(r, e);
}


DoubleDouble DoubleDouble::powi(int n) const
{
    int i = abs(n);
    DoubleDouble b = *this;
    DoubleDouble r(1);
    while (1) {
        if ((i & 1) == 1) {
            r = r * b;
        }
        if (i <= 1) {
            break;
        }
        i >>= 1;
        b = b*b;
    }
    if (n < 0) {
        return DoubleDouble(1) / r;
    }
    return r;
}


DoubleDouble DoubleDouble::exp() const
{
    int n = int(round(upper));
    DoubleDouble x(upper - n, lower);
    DoubleDouble u = (((((((((((x +
                     156)*x + 12012)*x +
                     600600)*x + 21621600)*x +
                     588107520)*x + 12350257920)*x +
                     201132771840)*x + 2514159648000)*x +
                     23465490048000)*x + 154872234316800)*x +
                     647647525324800)*x + 1295295050649600;
    DoubleDouble v = (((((((((((x -
                     156)*x + 12012)*x -
                     600600)*x + 21621600)*x -
                     588107520)*x + 12350257920)*x -
                     201132771840)*x + 2514159648000)*x -
                     23465490048000)*x + 154872234316800)*x -
                     647647525324800)*x + 1295295050649600;
    return dd_e.powi(n) * (u / v);
}

DoubleDouble DoubleDouble::sqrt() const
{
    double r = std::sqrt(upper);
    DoubleDouble sf = two_product(r, r);
    double e = (upper - sf.upper - sf.lower + lower) * 0.5 / r;
    return two_sum_quick(r, e);
}

DoubleDouble DoubleDouble::log() const
{
    DoubleDouble r(std::log(upper));
    DoubleDouble u = r.exp();
    r = r - DoubleDouble(2.0)*(u - *this)/(u + *this);
    return r;
}

#endif
