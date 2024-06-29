#include <vector>
#include <algorithm>
#include "xsimd/xsimd.hpp"


namespace minmax {

template<typename T>
struct minmax_pair {
    T min;
    T max;
};

//
// Simple scalar loop (i.e. not SIMD) to find the minimum and maximum
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
minmax_pair<T> minmax(const std::vector<T>& x)
{
    std::size_t size = x.size();
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
        auto arr_min = xsimd::reduce_min(minvec);
        auto arr_max = xsimd::reduce_max(maxvec);

        minmax_pair<T> tail_minmax = minmax_scalar_loop(size - i, &x[i]);

        auto xmin = std::min(arr_min, tail_minmax.min);
        auto xmax = std::max(arr_max, tail_minmax.max);

        return minmax_pair<T>{xmin, xmax};
    }
    else {
        return minmax_scalar_loop(x);
    }
}

}  // namespace minmax
