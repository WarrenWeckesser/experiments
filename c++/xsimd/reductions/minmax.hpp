#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>
#include "xsimd/xsimd.hpp"


namespace minmax {

template<typename T>
struct minmax_pair {
    T min;
    T max;
};

template<typename T>
struct value_index_pair {
    T value;
    std::size_t index;
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
        return minmax_scalar_loop(x);
    }
}

template<typename T>
value_index_pair<T> min_argmin_scalar_loop(const std::vector<T>& x)
{
    // x.size() must be at least 1.
    T xmin = x[0];
    size_t kmin = 0;
    for (size_t k = 1; k < x.size(); ++k) {
        if (x[k] < xmin) {
            xmin = x[k];
            kmin = k;
        }
    }
    return value_index_pair<T>{xmin, kmin};
}

template<typename T>
static inline void
vecs_to_arrays(xsimd::simd_type<T> value_vec,
               xsimd::simd_type<std::make_unsigned_t<xsimd::as_integer_t<T>>> index_vec,
               T value_arr[xsimd::simd_type<T>::size],
               std::size_t index_arr[xsimd::simd_type<T>::size],
               std::size_t index_offset = 0)
{
    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;

    xsimd::store(value_arr, value_vec);
    xsimd::store_as(index_arr, index_vec, xsimd::unaligned_mode());
    for (std::size_t i = 0; i < simd_size; ++i) {
        index_arr[i] = index_offset + index_arr[i] + i;
    }
}

//
// min_argmin
//
// Find the minimum element in x, return x and its index.
// If the minimum occurs more than once in x, the index of the first
// occurrence is returned.
//
// TO DO:
// * Clean up
// * Eliminate repeated code patterns (D.R.Y.)
// * Optimize where possible (surely there is room to improve this)
// * Why does the SIMD version for `double` give a speed up close to 2,
//   but for 64 bit integers there is no speed up?  For 64 bit integers,
//   the SIMD version is generally slower than the scalar loop.
//
template<typename T>
value_index_pair<T> min_argmin(const std::vector<T>& x)
{
    constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
    std::size_t size = x.size();
    std::size_t vec_size = size - size % simd_size;

    if (vec_size >= 2*simd_size) {
        T xmin;
        std::size_t kmin;

        using batch_int_t = std::make_unsigned_t<xsimd::as_integer_t<T>>;
        constexpr std::size_t batch_int_max = std::numeric_limits<batch_int_t>::max();
        constexpr std::size_t block_size = batch_int_max - batch_int_max % simd_size;

        if (size < block_size) {
            // The array can be processed in a single loop of SIMD operations.
            auto xminvec = xsimd::load_unaligned(&x[0]);
            xsimd::simd_type<batch_int_t> kminvec = xsimd::broadcast<batch_int_t>(0);
            for (std::size_t i = simd_size; i < vec_size; i += simd_size) {
                auto xvec = xsimd::load_unaligned(&x[i]);
                auto lt = xsimd::lt(xvec, xminvec);
                xminvec = xsimd::select(lt, xvec, xminvec);
                kminvec = xsimd::select(
                    xsimd::batch_bool_cast<batch_int_t>(lt),
                    xsimd::broadcast(static_cast<batch_int_t>(i)),
                    kminvec);
            }
            // If we interpret the input as a 2-d array with shape
            // (vec_size/simd_size, simd_size), then so far we have the minimum
            // value of each column in xminvec, and the row where it occurs
            // times simd_size in kminvec.  Now we have to get the overall
            // minimum value from xminvec, and compute its position in the
            // original 1-d input x.  Its position is the value from kminvec
            // plus the SIMD lane index.
            T xminv[simd_size];
            xsimd::store(xminv, xminvec);
            batch_int_t kminv[simd_size];
            xsimd::store(kminv, kminvec);
            T xm = xminv[0];
            std::size_t imin = 0;
            for (std::size_t i = 1; i < simd_size; ++i) {
                if (xminv[i] < xm) {
                    xm = xminv[i];
                    imin = i;
                }
            }
            xmin = xminv[imin];
            kmin = kminv[imin] + imin;

            // Process the final values of x, if there are any.
            if (vec_size != size) {
                T mm = x[vec_size];
                std::size_t imm = 0;
                for (std::size_t i = 1; i < size - vec_size; ++i) {
                    if (x[vec_size + i] < mm) {
                        mm = x[vec_size + i];
                        imm = i;
                    }
                }
                if (mm < xmin) {
                    kmin = vec_size + imm;
                }
            }
        }
        else {
            // Handle block_size < size...
            // XXX WIP -- Needs D.R.Y. clean up
            T xminarr[simd_size];
            std::size_t kminarr[simd_size];
            std::size_t block_start = 0;

            while (block_start < vec_size) {
                auto xminvec = xsimd::load_unaligned(&x[block_start]);
                xsimd::simd_type<batch_int_t> kminvec = xsimd::broadcast<batch_int_t>(0);
                std::size_t stop = block_start + block_size;
                if (stop > vec_size) {
                    stop = vec_size;
                }
                for (std::size_t i = block_start + simd_size; i < stop; i += simd_size) {
                    auto xvec = xsimd::load_unaligned(&x[i]);
                    auto lt = xsimd::lt(xvec, xminvec);
                    xminvec = xsimd::select(lt, xvec, xminvec);
                    kminvec = xsimd::select(
                        xsimd::batch_bool_cast<batch_int_t>(lt),
                        xsimd::broadcast(static_cast<batch_int_t>(i - block_start)),
                        kminvec);
                }
                if (block_start == 0) {
                    vecs_to_arrays(xminvec, kminvec, xminarr, kminarr);
                }
                else {
                    // Not the first block...
                    T xminarr1[simd_size];
                    std::size_t kminarr1[simd_size];
                    vecs_to_arrays(xminvec, kminvec, xminarr1, kminarr1, block_start);
                    // Merge (xminarr1, kminarr1) into (xminarr, kminarr)
                    for (std::size_t i = 0; i < simd_size; ++i) {
                        if (xminarr1[i] < xminarr[i]) {
                            xminarr[i] = xminarr1[i];
                            kminarr[i] = kminarr1[i];
                        }
                    }
                }
                block_start += block_size;
            }

            xmin = xminarr[0];
            kmin = kminarr[0];
            for (std::size_t i = 1; i < simd_size; ++i) {
                if (xminarr[i] < xmin) {
                    xmin = xminarr[i];
                    kmin = kminarr[i];
                }
            }

            // - - - - - - - - - - - - - - - - - - -

            if (vec_size != size) {
                T mm = x[vec_size];
                std::size_t imm = 0;
                for (std::size_t i = 1; i < size - vec_size; ++i) {
                    if (x[vec_size + i] < mm) {
                        mm = x[vec_size + i];
                        imm = i;
                    }
                }
                if (mm < xmin) {
                    kmin = vec_size + imm;
                }
            }
        }
        return value_index_pair<T>{xmin, kmin};
    }
    else {
        return min_argmin_scalar_loop(x);
    }
}

}  // namespace minmax
