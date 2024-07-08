
// For log and fabs used in pow_int64:
#include <math.h>

#include "int64_arithmetic.h"


int64_t
add_int64(int64_t a, int64_t b, int *perror)
{
    *perror = ARITHMETIC_OK;
    if (((a >= 0) && (b <= INT64_MAX - a)) || ((a < 0) && (b >= INT64_MIN - a))) {
        return a + b;
    }
    else {
        *perror = ARITHMETIC_OVERFLOW;
        return 0;
    }
}

int64_t
subtract_int64(int64_t a, int64_t b, int *perror)
{
    *perror = ARITHMETIC_OK;
    if (((b >= 0) && (a >= INT64_MIN + b)) || ((b < 0) && (a <= INT64_MAX + b))) {
        return a - b;
    }
    else {
        *perror = ARITHMETIC_OVERFLOW;
        return 0;
    }
}

int64_t
multiply_int64(int64_t a, int64_t b, int *perror)
{
    *perror = ARITHMETIC_OK;

    if (b == 0) {
        return 0;
    }
    if (b < 0) {
        b = -b;
        a = -a;
    }
    if ((INT64_MIN/b <= a) && (a <= INT64_MAX/b)) {
        return a*b;
    }
    else {
        *perror = ARITHMETIC_OVERFLOW;
        return 0;
    }
}

int64_t
pow_int64(int64_t b, int64_t p, int *perror)
{
    *perror = 0;
    if (p < 0) {
        *perror = -4;
        return 0;
    }
    if (p*log(fabs((double) b)) < log(INT64_MAX)) {
        return (p == 0) ? 1 : (p == 1) ? b : ((p % 2) ? b : 1) * pow_int64(b * b, p / 2, perror);
    }
    else {
        *perror = -1;
        return 0;
    }
}
