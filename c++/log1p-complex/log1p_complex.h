#ifndef LOG1P_COMPLEX_H
#define LOG1P_COMPLEX_H

#include <cstdio>
#include <cmath>
#include <complex>
#include <limits>
 
using namespace std;

template<typename T>
class doubled_t {
public:
    T upper;
    T lower;
};

//
// Dekker splitting.  See, for example, Theorem 1 of
//
//   Seppa Linnainmaa, Software for Double-Precision Floating-Point
//   Computations, ACM Transactions on Mathematical Software, Vol 7, No 3,
//   September 1981, pages 272-283.
//
// or Theorem 17 of
//
//   J. R. Shewchuk, Adaptive Precision Floating-Point Arithmetic and
//   Fast Robust Geometric Predicates, CMU-CS-96-140R, from Discrete &
//   Computational Geometry 18(3):305-363, October 1997.
//
template<typename T>
inline void
split(T x, doubled_t<T>& out)
{
    if (std::numeric_limits<T>::digits == 106) {
        // Special case: IBM double-double format.  The value is already
        // split in memory, so there is no need for any calculations.
        out.upper = static_cast<long double>(reinterpret_cast<double *>(&x)[0]);
        out.lower = static_cast<long double>(reinterpret_cast<double *>(&x)[1]);
    }
    else {
        constexpr int halfprec = (std::numeric_limits<T>::digits + 1)/2;
        T t = ((1ul << halfprec) + 1)*x;
        // The compiler must not be allowed to simplify this expression:
        out.upper = t - (t - x);
        out.lower = x - out.upper;
    }
}

template<typename T>
inline void
two_sum_quick(T x, T y, doubled_t<T>& out)
{
    T r = x + y;
    T e = y - (r - x);
    out.upper = r;
    out.lower = e;
}

template<typename T>
inline void
two_sum(T x, T y, doubled_t<T>& out)
{
    T s = x + y;
    T v = s - x;
    T e = (x - (s - v)) + (y - v);
    out.upper = s;
    out.lower = e;
}

template<typename T>
inline void
double_sum(const doubled_t<T>& x, const doubled_t<T>& y,
           doubled_t<T>& out)
{
    two_sum<T>(x.upper, y.upper, out);
    out.lower += x.lower + y.lower;
    two_sum_quick<T>(out.upper, out.lower, out);
}

template<typename T>
inline void
square(T x, doubled_t<T>& out)
{
    doubled_t<T> xsplit;
    out.upper = x*x;
    split(x, xsplit);
    // out.lower = (xsplit.upper*xsplit.upper - out.upper)
    //               + 2*xsplit.upper*xsplit.lower
    //               + xsplit.lower*xsplit.lower;
    out.lower = xsplit.lower*xsplit.lower
                - ((out.upper - xsplit.upper*xsplit.upper)
                   - 2*xsplit.lower*xsplit.upper);
}

//
// As the name makes clear, this function computes x**2 + 2*x + y**2.
// It uses doubled_t for the intermediate calculations.
//
// The function is used in log1p_complex() to avoid the loss of
// precision that can occur in the expression when x**2 + y**2 ≈ -2*x.
//
template<typename T>
inline T
xsquared_plus_2x_plus_ysquared(T x, T y)
{
    doubled_t<T> x2, y2, twox, sum1, sum2;

    square<T>(x, x2);               // x2 = x**2
    square<T>(y, y2);               // y2 = y**2
    twox.upper = 2*x;               // twox = 2*x
    twox.lower = 0.0;
    double_sum<T>(x2, twox, sum1);  // sum1 = x**2 + 2*x
    double_sum<T>(sum1, y2, sum2);  // sum2 = x**2 + 2*x + y**2
    return sum2.upper;
}

//
// For the float type, the intermediate calculation is done
// with the double type, instead of a doubled-float.
//
inline float
xsquared_plus_2x_plus_ysquared(float x, float y)
{
    double xd = x;
    double yd = y;
    return xd*(2.0 + xd) + yd*yd;
}


//
// Implement log1p(z) using double-double near the unit circle |z + 1| = 1
// to avoid loss of precision.
//
// This function assumes that neither part of z is nan.
//
// C is the complex type.
//
template<typename T>
complex<T>
log1p_complex(complex<T> z)
{
    T lnr;

    T x = z.real();
    T y = z.imag();
    if (x > -2.2 && x < 0.2 && y > -1.2 && y < 1.2) {
        // This nested `if` condition *should* be part of the outer
        // `if` condition, but clang on Mac OS 13 doesn't seem to
        // short-circuit the evaluation correctly and generates an
        // overflow when x and y are sufficiently large.
        if (fabs(x*(2.0 + x) + y*y) < 0.4) {
            // The input is close to the unit circle centered at -1+0j.
            // Use doubled_t to evaluate the real part of the result.
            // This is equivalent to 0.5*log((1+x)**2 + y**2), since
            //   log((1 + x)**2 + y**2) = log(1 + 2*x + x**2 + y**2)
            //                          = log1p(x**2 + 2*x + y**2)
            T t = xsquared_plus_2x_plus_ysquared<T>(x, y);
            lnr = 0.5*log1p(t);
        }
        else {
            lnr = log(hypot(x + 1, y));
        }
    }
    else {
        lnr = log(hypot(x + 1, y));
    }
    return complex<T>(lnr, atan2(y, x + 1.0));
}

#endif
