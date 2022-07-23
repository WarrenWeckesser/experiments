
#include <iostream>
#include <cstdint>
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

//
// Experimenting with swizzle and performing a min reduction of a batch.
// Currently the batches used here are hardcoded avx float (8 lanes).
//

//
// ShuffleLowHigh swizzle pattern generator.
//
// 8 lanes: 0 1 2 3 4 5 6 7 -> 0 4 1 5 2 6 3 7
// 4 lanes: 0 1 2 3         -> 0 2 1 3
//
struct ShuffleLowHigh
{
    static constexpr uint32_t get(size_t index, size_t size)
    {
        return (index & 1) ? size/2 + (index - 1)/2 : index/2;
    }
};

//
// SwapPairs swizzle pattern generator.
//
// Swap 0 & 1, 2 & 3, etc.
//
struct SwapPairs
{
    static constexpr uint32_t get(size_t index, size_t size)
    {
        return index ^ 1;
    }
};


int main(int argc, char* argv[])
{
    xs::batch<float, xs::avx> a = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    std::cout << "a: " << a << std::endl;

    constexpr auto shuffle_low_high = xs::make_batch_constant<xs::batch<std::uint32_t,
                                                                        xs::avx>,
                                                              ShuffleLowHigh>();

    auto b = xs::swizzle(a, shuffle_low_high);
    std::cout << "b: " << b << " ShuffleLowHigh" << std::endl;

    constexpr auto swap_pairs = xs::make_batch_constant<xs::batch<std::uint32_t,
                                                                  xs::avx>,
                                                        SwapPairs>();
    auto c = xs::swizzle(a, swap_pairs);
    std::cout << "c: " << c << " SwapPairs" << std::endl;

    std::cout << std::endl;

    xs::batch<float, xs::avx> x = {3.0, 1.0, -2.0, 2.5, -1.0, 4.0, 3.5, 2.0};
    std::cout << "x: " << x << std::endl;

    // With 8 lanes, we need to repeat the sequence of swap+min and shuffle+min
    // three times.  (In general, it will be log2(N) repetitions, where N is the
    // number of lanes.)
    auto y = xs::min(x, xs::swizzle(x, swap_pairs));
    y = xs::min(y, xs::swizzle(y, shuffle_low_high));
    y = xs::min(y, xs::swizzle(y, swap_pairs));
    y = xs::min(y, xs::swizzle(y, shuffle_low_high));
    y = xs::min(y, xs::swizzle(y, swap_pairs));
    y = xs::min(y, xs::swizzle(y, shuffle_low_high));
    
    std::cout << "y: " << y << std::endl;

    float xbuf[x.size];
    y.store_unaligned(xbuf);
    std::cout << "min(x) = " << xbuf[0] << std::endl;

    return 0;
}
