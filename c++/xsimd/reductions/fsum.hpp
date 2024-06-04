// XXX In progress...

#include <vector>
#include "xsimd/xsimd.hpp"

using namespace std;

namespace fsum {

//
// Simple scalar loop (i.e. not SIMD) to compute the fsum
// of an array.  A single pass is made through the array.
//
// size must be at least 1.
//

#pragma GCC push_options
#pragma GCC optimize ("no-fast-math")

template<typename T>
T fsum_scalar_loop(const vector<T>& x)
{
    T sum = 0.0;
    T c = 0.0;
    for (const auto &x1 : x) {
        T y = x1 - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template<typename T>
T fsum_scalar_loop(std::size_t n, const T *x)
{
    T sum = 0.0;
    T c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        T y = x[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

#pragma GCC pop_options


//
// XXX FIXME:
// This templated function assumes that T will be a type that can
// be used in an XSIMD batch.
//
template<typename T>
T fsum(const vector<T>& x)
{
    size_t size = x.size();
    constexpr size_t nlanes = xsimd::simd_type<T>::size;
    size_t vec_size = size - size % nlanes;

    if (vec_size >= 2*nlanes) {
        auto sum = xsimd::batch<T>{0.0};
        auto c = xsimd::batch<T>{0.0};
        size_t i;
        for (i = 0; i < vec_size; i += nlanes) {
            auto y = xsimd::load_unaligned(&x[i]);
            auto t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        T vec[nlanes];
        xsimd::store(&vec[0], sum);
        T vecsum = fsum_scalar_loop(nlanes, vec);
        if (size % nlanes) {
            // XXX Not a compensated sum.
            vecsum += fsum_scalar_loop(size % nlanes, &x[i]);
        }
        return vecsum;
    }
    else {
        return fsum_scalar_loop(x);
    }
}

}  // namespace fsum
