// XXX No one should use this code as an example of
//     *good* SIMD programming!

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
    size_t k = 0;
    for (; k < m - r; k += 4) {
        __m128 z4 = _mm_loadu_ps(z + k);
        //print_m128("z4:  ", z4);
        __m128 s1 = _mm_mul_ps(z4, z4);
        //print_m128("s1:  ", s1);
        __m128 s2 = _mm_movehdup_ps(s1);
        //print_m128("s2:  ", s2);
        __m128 ss = _mm_add_ps(s1, s2);
        //print_m128("ss:  ", ss);
        __m128 as2 = _mm_shuffle_ps(ss, ss, _MM_SHUFFLE(1,3,2,0));
        //print_m128("as2: ", as2);
        _mm_store_sd((double *) (out+k/2), _mm_castps_pd(as2));
    }
    if (r > 0) {
        cabs2f(r/2, z + k, out + k/2);
    }
    return 0;
}

int cabs2f_avx(size_t n, const float *z, float *out)
{
    __m256i idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    __m256i storemask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);

    size_t r = n % 8;
    size_t k = 0;
    for (; k < n - r; k += 8) {
        __m256 z8 = _mm256_loadu_ps(z + k);
        //print_m256("z8: ", z8);
        __m256 s1 = _mm256_mul_ps(z8, z8);
        //print_m256("s1: ", s1);
        // XXX FIXME: Replace 177 with the appropriate macro.
        __m256 s2 = _mm256_permute_ps(s1, 177);
        //print_m256("s2: ", s2);
        __m256 ss = _mm256_add_ps(s1, s2);
        //print_m256("ss: ", ss);
        __m256 as = _mm256_permutevar8x32_ps(ss, idx);
        //print_m256("as: ", as);
        _mm256_maskstore_ps(out + k/2, storemask, as);
    }
    cabs2f(r, z + k, out + k/2);
    return 0;
}
