#ifndef MIN_MAX_H
#define MIN_MAX_H

#include <stddef.h>

float float_min_keep_nan(size_t, float *);
float float_min(size_t, float *);
float float_min_sse(size_t, float *);
float float_min_avx(size_t, float *);

float float_max_keep_nan(size_t, float *);
float float_max(size_t, float *);
float float_max_sse(size_t, float *);
float float_max_avx(size_t, float *);

#endif
