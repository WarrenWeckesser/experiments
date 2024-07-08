#ifndef INT64_ARITHMETIC_H
#define INT64_ARITHMETIC_H

#include <stdint.h>

#define ARITHMETIC_OK        0
#define ARITHMETIC_OVERFLOW -1

int64_t
add_int64(int64_t a, int64_t b, int *perror);

int64_t
subtract_int64(int64_t a, int64_t b, int *perror);

int64_t
multiply_int64(int64_t a, int64_t b, int *perror);

int64_t
pow_int64(int64_t b, int64_t p, int *perror);

#endif
