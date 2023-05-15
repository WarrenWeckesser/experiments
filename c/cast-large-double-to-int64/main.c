#include <stdio.h>
#include <inttypes.h>

int main(int argc, char *argv[])
{
    // Casting a floating point value to an int when the value exceeds
    // the maximum of the integer type is undefined behavior.  The result
    // is compiler-dependent.

    double x = 9223372036854775808.0;
    int64_t i = (int64_t) x;

    printf("x = %lf\n", x);
    printf("i = %" PRId64 "\n", i);

    return 0;
}
