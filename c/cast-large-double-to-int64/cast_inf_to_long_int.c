#include <stdio.h>
#include <inttypes.h>
#include <math.h>

int main(int argc, char *argv[])
{
    double x = -INFINITY;
    long i = (long) x;

    printf("x = %lf\n", x);
    printf("i = %ld\n", i);

    return 0;
}
