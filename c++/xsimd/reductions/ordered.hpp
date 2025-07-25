#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>
#include "xsimd/xsimd.hpp"

#include <iostream>


namespace ordered {

//
// Simple scalar loop (i.e. not SIMD) to determine if a 1-d array has
// increasing values.  Array with length 0 and 1 are considered to be
// trivially increasing.
//
template<typename T>
bool is_increasing_scalar_loop(std::size_t size, const T* x)
{
    for (std::size_t i = 1; i < size; ++i) {
        if (x[i] < x[i - 1]) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool is_increasing_scalar_loop(const std::vector<T>& x)
{
    return is_increasing_scalar_loop(x.size(), &x[0]);
}

template<typename T>
bool is_increasing_v1(const std::vector<T>& x)
{
    std::size_t size = x.size();
    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
    std::size_t vec_size = size - (size - 1) % simd_size;

    if (vec_size >= 2*simd_size) {
        auto i = simd_size;
        for (i = 1; i < vec_size; i += simd_size) {
            auto vec0 = xsimd::load_unaligned(&x[i-1]);
            auto vec1 = xsimd::load_unaligned(&x[i]);  // XXX More memory access than necessary?
            auto cmp = xsimd::gt(vec0, vec1);
            if (xsimd::any(cmp)) {
                return false;
            }
        }
        if (vec_size != size) {
            return is_increasing_scalar_loop(size - vec_size + 1, &x[vec_size - 1]);
        }

        return true;
    }
    else {
        return is_increasing_scalar_loop(x);
    }
}


//
// v2 avoids reading the same values from memory twice, but it is
// slower than v2.  Presumably, in v1, the second read is fast because
// the values to be read are in the cache.  v2 has overhead coming from
// the shuffle function.
//
template<typename T>
bool is_increasing_v2(const std::vector<T>& x)
{
    std::size_t size = x.size();
    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
    std::size_t vec_size = size - (size % simd_size);

    if (vec_size >= 2*simd_size) {
        auto i = simd_size;
        auto x0 = xsimd::load_unaligned(&x[0]);
        for (i = simd_size; i < vec_size; i += simd_size) {

            struct slide2
            {
                static constexpr unsigned get(unsigned i, [[maybe_unused]] unsigned n)
                {
                    return i + 1;
                }
            };

            auto x1 = xsimd::load_unaligned(&x[i]);
            auto y = xsimd::shuffle(x0, x1,
                                    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>,
                                                               slide2>());
            auto cmp = xsimd::gt(x0, y);
            if (xsimd::any(cmp)) {
                return false;
            }
            x0 = x1;
        }
        return is_increasing_scalar_loop(size - vec_size + simd_size, &x[vec_size - simd_size]);

        return true;
    }
    else {
        return is_increasing_scalar_loop(x);
    }
}

}  // namespace ordered
