#include <stdio.h>
#include <math.h>

struct _doubledouble_t {
    double upper;
    double lower;
};

typedef struct _doubledouble_t doubledouble_t;


//
// Dekker splitting.  See, for example, Theorem 1 of
//
//   Seppa Linnainmaa, Software for Double-Precision Floating-Point
//   Computations, ACM Transactions on Mathematical Software, Vol 7, No 3,
//   September 1981, pages 272-283.
//
static void
split(double x, doubledouble_t *out)
{
    double t = ((1 << 27) + 1)*x;
    out->upper = t - (t - x);
    out->lower = x - out->upper;
}

static void
split_fma(double x, doubledouble_t *out)
{
    double t = ((1 << 27) + 1)*x;
    out->upper = fma(-(1 << 27), x, t);
    out->lower = x - out->upper;
}

int main(int argc, char *argv[])
{
    doubledouble_t out;

    double x = 0.125000000001;

#ifdef FP_FAST_FMA
    printf("FP_FAST_FMA is defined.\n");
#endif

    split(x, &out);
    printf("%25.17e  %25.17e\n", out.upper, out.lower);

    split_fma(x, &out);
    printf("%25.17e  %25.17e\n", out.upper, out.lower);

    return 0;
}
