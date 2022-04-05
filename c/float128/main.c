#include <stdio.h>
#include <quadmath.h>

void
print_float128(__float128 x)
{
    char buf[1000];
    int width = 62;

    quadmath_snprintf(buf, sizeof(buf), "%+-#*.32Qe", width, x);
    printf("%s\n", buf);
}

void
print_diff(const char *name, __float128 xq, double x)
{
    char buf[1000];
    int width = 54;

    __float128 diff = xq - x;
    quadmath_snprintf(buf, sizeof(buf), "%+-#*.26Qe", width, diff);
    printf("%s diff = %s\n", name, buf);
}

int main(int argc, char *argv[])
{
    __float128 piq = M_PIq;
    double pi = 0x1.921fb54442d18p+1;
    print_diff("pi", piq, pi);

    __float128 eq = M_Eq;
    double e = 0x1.5bf0a8b145769p+1;
    print_diff("e ", eq, e);

    __float128 exppi = expq(piq);
    printf("exp(pi) = ");
    print_float128(exppi);

    // A check for a result from a double-double calculation of exp(pi).
    double high = 23.14069263277927035;
    double low = -1.34887470919957884e-15;
    __float128 highq = high;
    __float128 lowq = low;
    __float128 check = highq + lowq;
    printf("check   = ");
    print_float128(check);

    return 0;
}
