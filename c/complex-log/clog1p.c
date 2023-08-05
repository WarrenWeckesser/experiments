
#include <complex.h>
#include <math.h>

double complex
clog1p(double complex z)
{
    double complex w;

    if (isnan(creal(z)) || isnan(cimag(z))) {
        w = CMPLX(NAN, NAN);
    }
    else {
        double _Complex u = z + 1.0;
        if (creal(u) == 1.0 && cimag(u) == 0.0) {
            // z + 1 == 1
            w = z;
        }
        else {
            if (creal(u) - 1.0 == creal(z)) {
                // u - 1 == z
                w = clog(u);
            }
            else {
                w = clog(u) * (z / (u - 1.0));
            }
        }
    }
    return w;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Double-double functions used by log1p to avoid loss of precision
// when the complex input z is close to the unit circle centered at -1+0j.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

struct _doubledouble_t {
    double upper;
    double lower;
};

typedef struct _doubledouble_t doubledouble_t;

static void
two_sum_quick(double x, double y, doubledouble_t *out)
{
    double r = x + y;
    double e = y - (r - x);
    out->upper = r;
    out->lower = e;
}

static void
two_sum(double x, double y, doubledouble_t *out)
{
    double s = x + y;
    double v = s - x;
    double e = (x - (s - v)) + (y - v);
    out->upper = s;
    out->lower = e;
}

static void
double_sum(const doubledouble_t x, const doubledouble_t y,
           doubledouble_t *out)
{
    two_sum(x.upper, y.upper, out);
    out->lower += x.lower + y.lower;
    two_sum_quick(out->upper, out->lower, out);
}

static void
split(double x, doubledouble_t *out)
{
    double t = ((1 << 27) + 1)*x;
    out->upper = t - (t - x);
    out->lower = x - out->upper;
}

static void
square(double x, doubledouble_t *out)
{
    doubledouble_t xsplit;
    out->upper = x*x;
    split(x, &xsplit);
    out->lower = (xsplit.upper*xsplit.upper - out->upper)
                  + 2*xsplit.upper*xsplit.lower
                  + xsplit.lower*xsplit.lower;
}

static double
foo(double x, double y)
{
    doubledouble_t x2, y2, twox, sum1, sum2;

    square(x, &x2);
    square(y, &y2);
    twox.upper = 2*x;
    twox.lower = 0.0;
    double_sum(x2, twox, &sum1);
    double_sum(sum1, y2, &sum2);
    return sum2.upper;
}

complex double
clog1p_doubledouble(complex double z)
{
    double complex w;

    double x = creal(z);
    double y = cimag(z);

    if (isnan(x) || isnan(y)) {
        w = CMPLX(NAN, NAN);
    }
    else {
        double lnr;
        if (x > -2.2 && x < 0.2 && y > -1.2 && y < 1.2
                && fabs(x*(2.0 + x) + y*y) < 0.4) {
            // The input is close to the unit circle centered at -1+0j.
            // Use double-double to evaluate the real part of the result.
            lnr = 0.5*log1p(foo(x, y));
        }
        else {
            lnr = log(hypot(x + 1, y));
        }
        w = CMPLX(lnr, atan2(y, x + 1.0));
    }
    return w;
}
