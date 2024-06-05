
#include <stdio.h>
#include <fenv.h>
#include <math.h>
#include <errno.h>

void show_fp_exception_flags()
{
    if (fetestexcept(FE_DIVBYZERO)) {
        printf(" FE_DIVBYZERO");
    }
    // FE_INEXACT is common and not interesting.
    // if (std::fetestexcept(FE_INEXACT)) {
    //     cout << " FE_INEXACT";
    // }
    if (fetestexcept(FE_INVALID)) {
        printf(" FE_INVALID");
    }
    if (fetestexcept(FE_OVERFLOW)) {
        printf(" FE_OVERFLOW");
    }
    if (fetestexcept(FE_UNDERFLOW)) {
        printf(" FE_UNDERFLOW");
    }
    printf("\n");
}

int main()
{
    double x = -INFINITY;

    feclearexcept(FE_ALL_EXCEPT);
    errno = 0;

    double y = log(x);
    show_fp_exception_flags();

    printf("log(x) = %lf\n", y);
    printf("errno = %d\n", errno);

}
