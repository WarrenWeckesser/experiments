
#include <cstdio>
#include "doubledouble.h"


void
print_doubledouble(const char *prefix, const DoubleDouble& x)
{
    printf("%s = (%24.17f, %24.17e)\n", prefix, x.upper, x.lower);    
}


int main(int argc, char *argv[])
{
    auto x = DoubleDouble(3.0);
    auto y = DoubleDouble(1.5, 2.5e-18);

    print_doubledouble("x", x);
    print_doubledouble("y", y);

    auto z = x + y;

    print_doubledouble("   x + y", z);

    z = x - y;
    print_doubledouble("   x - y", z);

    z = -x + y;
    print_doubledouble("  -x + y", z);

    z = x * y;
    print_doubledouble("   x * y", z);

    z = x / y;
    print_doubledouble("   x / y", z);

    z = y / x;
    print_doubledouble("   y / x", z);

    z = DoubleDouble(1.0) / x;
    print_doubledouble("   1 / x", z);

    z = y.powi(3);
    print_doubledouble("    y**3", z);

    z = y.exp();
    print_doubledouble("  exp(y)", z);

    z = dd_pi.exp();
    print_doubledouble(" exp(pi)", z);

    z = dd_pi.log();
    print_doubledouble(" log(pi)", z);

    z = dd_pi.sqrt();
    print_doubledouble("sqrt(pi)", z);
    return 0;
}
