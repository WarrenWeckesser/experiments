
#include <cstdio>
#include <cmath>
#include <cfenv>

int main()
{
    double x = 1e200;
    double y = 1e200;

    std::feclearexcept(FE_ALL_EXCEPT);
    if (x < 2 && x > -2 && y < 2 && y > -2 && std::fabs(x*(2+x) + y*y) < 0.4) {
        printf("Boo!\n");
    }
    else {
        printf("Condition is false.\n");
    }
    if (std::fetestexcept(FE_OVERFLOW)) {
        printf("FE_OVERFLOW is set! (1)\n");
    }
    double z = std::fabs(x*(2+x) + y*y);
    if (std::fetestexcept(FE_OVERFLOW)) {
        printf("FE_OVERFLOW is set! (2)\n");
    }
    printf("z = %lf\n", z);
}
