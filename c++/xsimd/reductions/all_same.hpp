#include "xsimd/xsimd.hpp"
#include <vector>

namespace all_same
{

//
// Simple scalar loop (i.e. not SIMD) to determine if all the values
// in the input array x are the same as the given value v.
//
template <typename T>
bool
all_same_scalar_loop(std::size_t size, const T v, const T *x)
{
    if (size > 0) {
        for (std::size_t i = 0; i < size; ++i) {
            if (x[i] != v) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool
all_same_scalar_loop(const T v, const std::vector<T> &x)
{
    return all_same_scalar_loop(x.size(), v, &x[0]);
}

template <typename T>
bool
all_same(const std::vector<T> &x)
{
    std::size_t size = x.size();
    if (size == 0) {
        return true;  // Trivially true
    }

    T x0 = x[0];

    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
    std::size_t vec_size = size - size % simd_size;

    if (vec_size >= 2 * simd_size) {
        auto x0vec = xsimd::broadcast(x0);
        for (std::size_t i = 0; i < vec_size; i += simd_size) {
            auto vec = xsimd::load_unaligned(&x[i]);
            if (xsimd::count(xsimd::neq(vec, x0vec))) {
                return false;
            }
        }
        if (vec_size != size) {
            std::size_t rem = size - vec_size;
            return all_same_scalar_loop(rem, x0, &x[size - rem]);
        }
        return true;
    } else {
        return all_same_scalar_loop(x0, x);
    }
}

}  // namespace all_same
