#ifndef CABS2_H
#define CABS2_H

int cabs2f(size_t n, const float *z, float *out);
int cabs2f_sse(size_t n, const float *z, float *out);
int cabs2f_avx(size_t n, const float *z, float *out);

#endif
