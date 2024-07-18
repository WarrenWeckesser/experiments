#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "xsimd/xsimd.hpp"


namespace minmax {

template<typename T>
struct minmax_pair {
    T min;
    T max;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const minmax_pair<T>& obj)
{
    os << '{' << +obj.min << ", " << +obj.max << '}';
    return os;
}

//
// Simple scalar loop (not SIMD) to find the minimum and maximum
// of an array.  A single pass is made through the array.
//
// size must be at least 1.
//
template<typename T>
minmax_pair<T> minmax_scalar_loop(std::size_t size, const T* x)
{
    auto xmin = x[0];
    auto xmax = x[0];
    for (std::size_t i = 1; i < size; ++i) {
        xmin = std::min(xmin, x[i]);
        xmax = std::max(xmax, x[i]);
    }
    return minmax_pair<T>{xmin, xmax};
}

template<typename T>
minmax_pair<T> minmax_scalar_loop(const std::vector<T>& x)
{
    return minmax_scalar_loop(x.size(), &x[0]);
}


template<typename T>
minmax_pair<T> minmax_unaligned(std::size_t size, T* x)
{
    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
    std::size_t vec_size = size - size % simd_size;

    if (vec_size >= 2*simd_size) {
        auto minvec = xsimd::load_unaligned(&x[0]);
        auto maxvec = minvec;
        auto i = simd_size;
        for (; i < vec_size; i += simd_size) {
            auto vec = xsimd::load_unaligned(&x[i]);
            minvec = xsimd::min(minvec, vec);
            maxvec = xsimd::max(maxvec, vec);
        }
        if (vec_size != size) {
            auto vec = xsimd::load_unaligned(&x[size - simd_size]);
            minvec = xsimd::min(minvec, vec);
            maxvec = xsimd::max(maxvec, vec);
        }
        auto arr_min = xsimd::reduce_min(minvec);
        auto arr_max = xsimd::reduce_max(maxvec);
        return minmax_pair<T>{arr_min, arr_max};
    }
    else {
        return minmax_scalar_loop(size, x);
    }
}

template<typename T>
minmax_pair<T> minmax_try_aligned(std::size_t size, T* x)
{
    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
    std::size_t space = size * sizeof(T);
    std::size_t space_after_align = space;
    void *ptr = x;
    const size_t min_batches = 2;
    void *p = std::align(sizeof(xsimd::simd_type<T>),
                         min_batches*sizeof(xsimd::simd_type<T>),
                         ptr, space_after_align);
    if (!p) {
        return minmax_unaligned(size, x);
    }
    // By how many bytes did the pointer change:
    size_t delta = space - space_after_align;
    if (delta % sizeof(T) != 0) {
        // Unusual alignment.
        return minmax_unaligned(size, x);
    }
    std::size_t nskip = delta / sizeof(T);

    xsimd::simd_type<T> minvec;
    xsimd::simd_type<T> maxvec;

    if (nskip > 0) {
        // First element of the array is not aligned.
        // This vector will overlap with the first aligned vector
        // by simd_size - nskip lanes.  The overlap is not a problem
        // for min and max.
        minvec = xsimd::load_unaligned(&x[0]);
    }
    else {
        minvec = xsimd::load_aligned(&x[0]);
    }
    maxvec = minvec;

    std::size_t reduced_size = size - nskip;
    std::size_t vec_size = reduced_size - reduced_size % simd_size;

    auto i = simd_size;
    for (; i < vec_size; i += simd_size) {
        auto vec = xsimd::load_aligned(&x[nskip + i]);
        minvec = xsimd::min(minvec, vec);
        maxvec = xsimd::max(maxvec, vec);
    }
    size_t ntail = reduced_size - i;
    if (ntail > 0) {
        auto vec = xsimd::load_unaligned(&x[size - simd_size]);
        minvec = xsimd::min(minvec, vec);
        maxvec = xsimd::max(maxvec, vec);
    }
    auto xmin = xsimd::reduce_min(minvec);
    auto xmax = xsimd::reduce_max(maxvec);

    return minmax_pair<T>{xmin, xmax};
}

}  // namespace minmax
