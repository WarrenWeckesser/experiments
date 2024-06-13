#include <stdio.h>
#include <quadmath.h>

int main()
{
    long double x = -0.2L;
    printf("x = %32.25Lf\n", x);
    __float128 y = x;
    x = y;
    printf("x = %32.25Lf\n", x);
    
    return 0;
}