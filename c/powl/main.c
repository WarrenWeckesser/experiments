
#include <stdio.h>
#include <math.h>
#include <fenv.h>

#if defined(__APPLE__)
#define FOO "APPLE\n"
#else
#define FOO "Something else\n"
#endif

long double my_powl(long double x, long double y)
{
    if (x == 0 && y > 0) {
        return 0.0L;
    }
    return powl(x, y);
}


int main(int argc, char *argv[])
{
    long double x, y, z;
    int e;

    printf(FOO);

    x = 0.0L;
    y = 2.5L;
    feclearexcept(FE_ALL_EXCEPT);
    z = powl(x, y);
    e = fetestexcept(FE_ALL_EXCEPT);
    printf("e = %d\n", e);

    feclearexcept(FE_ALL_EXCEPT);
    z = my_powl(x, y);
    e = fetestexcept(FE_ALL_EXCEPT);
    printf("e = %d\n", e);

    return 0;
}
