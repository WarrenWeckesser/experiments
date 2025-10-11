#include <iostream>
#include <cstdint>
#include "xsimd/xsimd.hpp"

namespace xs = xsimd;

//
// Experimenting with swizzle.
//

//
// ShuffleLowHigh swizzle pattern generator.
//
// 8 lanes: 0 1 2 3 4 5 6 7 -> 0 4 1 5 2 6 3 7
// 4 lanes: 0 1 2 3         -> 0 2 1 3
//
struct ShuffleLowHigh
{
    static constexpr size_t get(size_t index, size_t size)
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
    static constexpr size_t get(size_t index, size_t size)
    {
        return index ^ 1;
    }
};


template<size_t n>
struct ShiftLeft
{
    static constexpr size_t get(size_t index, size_t size)
    {
        return (index + n) % size;
    }
};


int main(int argc, char* argv[])
{
    using BatchType = xs::batch<float>;
    using LaneType = BatchType::value_type;

    constexpr std::size_t nlanes = BatchType::size;

    // Create batch `a` containing {0.0, 1.0, 2.0, ...}
    std::array<LaneType, nlanes> avals;
    for (std::size_t i = 0; i < nlanes; ++i) {
        avals[i] = i;
    }
    BatchType a = xs::load_unaligned(avals.data());

    std::cout << "a:  " << a << std::endl;

    constexpr auto shuffle_low_high = xs::make_batch_constant<xs::as_unsigned_integer_t<LaneType>,
                                                              ShuffleLowHigh>();

    auto b = xs::swizzle(a, shuffle_low_high);
    std::cout << "b:  " << b << " ShuffleLowHigh" << std::endl;

    constexpr auto swap_pairs = xs::make_batch_constant<xs::as_unsigned_integer_t<LaneType>,
                                                        SwapPairs>();
    auto c = xs::swizzle(a, swap_pairs);
    std::cout << "c:  " << c << " SwapPairs" << std::endl;

    constexpr auto shift1 = xs::make_batch_constant<xs::as_unsigned_integer_t<LaneType>,
                                                    ShiftLeft<1>>();
    constexpr auto shift2 = xs::make_batch_constant<xs::as_unsigned_integer_t<LaneType>,
                                                    ShiftLeft<2>>();

    auto s1 = xs::swizzle(a, shift1);
    auto s2 = xs::swizzle(a, shift2);
    std::cout << "s1: " << s1 << " ShiftLeft<1>" << std::endl;
    std::cout << "s2: " << s2 << " ShiftLeft<2>" << std::endl;

    if (nlanes > 4) {
        constexpr auto shift4 = xs::make_batch_constant<xs::as_unsigned_integer_t<LaneType>,
                                                        ShiftLeft<4>>();
        auto s4 = xs::swizzle(a, shift4);
        std::cout << "s4: " << s4 << " ShiftLeft<4>" << std::endl;
    }

    std::cout << std::endl;

    return 0;
}
