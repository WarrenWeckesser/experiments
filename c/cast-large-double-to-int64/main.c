#include <stdio.h>
#include <inttypes.h>

int main(int argc, char *argv[])
{
    double x = 9223372036854775808.0;
    int64_t i = (int64_t) x;

    printf("x = %lf\n", x);
    printf("i = %" PRId64 "\n", i);

    return 0;
}
