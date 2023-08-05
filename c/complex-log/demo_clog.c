
#include <stdio.h>
#include <complex.h>
#include <math.h>

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Double-double functions used by my_clog to avoid loss of precision
// when the complex input z is close to the unit circle centered at 0.
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

static void
sumsq(double x, double y, doubledouble_t *sum)
{
    doubledouble_t x2, y2;

    square(x, &x2);
    square(y, &y2);
    double_sum(x2, y2, sum);
}

//
// This version of the complex log(z) does extra work when |z| is near 1
// to ensure that the real part of the result is very accurate.
//
double complex
my_clog(double complex z)
{
    double lnr;

    double x = creal(z);
    double y = cimag(z);

    if (x == 0 && y == 0) {
        return CMPLX(-INFINITY, 0.0);
    }
    double theta = atan2(y, x);

    double r = hypot(x, y);
    // The following thresholds were not chosen carefully!
    if (r > 0.9 && r < 1.1) {
        doubledouble_t sum;
        // Use double-double to (re)compute x**2 + y**2 when the point
        // is near the unit circle, to avoid the loss of precision that
        // occurs when the log of the sum of squares is taken.
        sumsq(x, y, &sum);
        // We know sum.upper > 0, and sum.upper >> |sum.lower|. Use
        //   log(upper + lower) = log(upper*(1 + lower/upper))
        //                      = log(upper) + log1p(lower/upper)
        lnr = 0.5*(log(sum.upper) + log1p(sum.lower/sum.upper));
    }
    else {
        lnr = log(r);
    }

    return CMPLX(lnr, theta);
}

int main(int argc, char *argv[])
{
    double complex z[] = {
        CMPLX(0.5469835951544786, 0.8371433277465153),
        CMPLX(0.9689124217106447, -0.24740395925452294),
        CMPLX(-0.9958083245390612, 0.0914646422324372)
    };

    for (int k = 0; k < sizeof(z)/sizeof(z[0]); ++k) {
        printf("z          = %25.17e + %25.17ej  cabs(z) = %25.17e\n",
               creal(z[k]), cimag(z[k]), cabs(z[k]));

        double complex w1 = clog(z[k]);
        printf("clog(z)    = %25.17e + %25.17ej\n", creal(w1), cimag(w1));

        double complex w2 = my_clog(z[k]);
        printf("my_clog(z) = %25.17e + %25.17ej\n", creal(w2), cimag(w2));

        printf("\n");
    }

    return 0;
}
