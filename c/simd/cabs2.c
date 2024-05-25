#include <math.h>
#include <immintrin.h>
#include <stdio.h>


void
print_m128(const char *prefix, __m128 x)
{
    float data[4];

    printf("%s", prefix);
    _mm_store_ps(data, x);
    for (size_t k = 0; k < 4; ++k) {
        printf(" %12.8f", data[k]);
    }
    printf("\n");
}


void
print_m256(const char *prefix, __m256 x)
{
    float data[8];

    printf("%s", prefix);
    _mm256_storeu_ps(data, x);
    for (size_t k = 0; k < 8; ++k) {
        printf(" %12.8f", data[k]);
    }
    printf("\n");
}

//
// z must be an array of length 2*n.  Each consecutive pair
// of floats represents a complex number.
//
// out must be an array of length n.  The squared modulus
// of each complex number in z is stored in out.  

int cabs2f(size_t n, const float *z, float *out)
{
    for (size_t k = 0; k < n; ++k) {
        out[k] = z[2*k]*z[2*k] + z[2*k+1]*z[2*k+1];
    }
    return 0;
}

int cabs2f_sse(size_t n, const float *z, float *out)
{
    size_t m = 2*n;
    size_t r = m % 4;
    size_t k;
    for (k = 0; k < m - r; k += 4) {
        __m128 z4 = _mm_loadu_ps(z + k);
        __m128 s1 = _mm_mul_ps(z4, z4);
        __m128 s2 = _mm_movehdup_ps(s1);
        __m128 ss = _mm_add_ps(s1, s2);
        __m128 as2 = _mm_shuffle_ps(ss, ss, _MM_SHUFFLE(1,3,2,0));
        _mm_store_sd((double *) (out+k/2), _mm_castps_pd(as2));
    }
    if (r > 0) {
        cabs2f(r/2, z + k, out + k/2);
    }
    return 0;
}

int cabs2f_sse_v2(size_t n, const float *z, float *out)
{
    size_t m = 2*n;
    size_t r = m % 8;
    size_t k;
    for (k = 0; k < m - r; k += 8) {
        __m128 z1 = _mm_loadu_ps(z + k);
        __m128 z2 = _mm_loadu_ps(z + k + 4);
        __m128 s1 = _mm_mul_ps(z1, z1);
        __m128 s2 = _mm_mul_ps(z2, z2);
        __m128 s1_mix = _mm_shuffle_ps(s1, s2, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 s2_mix = _mm_shuffle_ps(s1, s2, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 ss_mix = _mm_add_ps(s1_mix, s2_mix);
        _mm_store_ps(out + k/2, ss_mix);
    }
    if (r > 0) {
        cabs2f(r/2, z + k, out + k/2);
    }
    return 0;
}

int cabs2f_avx(size_t n, const float *z, float *out)
{
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    size_t m = 2*n;
    size_t r = m % 16;
    size_t k;
    // This loop processes 8 complex numbers at a time.
    for (k = 0; k < m - r; k += 16) {
        // Load 4 complex numbers into z1, and the next 4 into z2.
        __m256 z1 = _mm256_loadu_ps(z + k);
        __m256 z2 = _mm256_loadu_ps(z + k + 8);

        // s1 and s2 hold the squared components of the complex numbers.
        __m256 s1 = _mm256_mul_ps(z1, z1);
        __m256 s2 = _mm256_mul_ps(z2, z2);

        // Get all the squared x (i.e. real) components into xx_mix, and the
        // squared y (i.e. imaginary) components in to yy_mix.  The values
        // from the same input complex number are aligned within xx_mix and
        // yy_mix, but they are not in the same order as in the input array z.
        __m256 xx_mix = _mm256_shuffle_ps(s1, s2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 yy_mix = _mm256_shuffle_ps(s1, s2, _MM_SHUFFLE(3, 1, 3, 1));

        // Add the squared components.
        __m256 ss_mix = _mm256_add_ps(xx_mix, yy_mix);

        // Permute the sum of squares in ss_mix back to the correct order.
        __m256 ss = _mm256_permutevar8x32_ps(ss_mix, idx);

        _mm256_store_ps(out + k/2, ss);
    }
    // Compute the result for the remaining values with the non-SIMD function.
    cabs2f(r/2, z + k, out + k/2);
    return 0;
}
