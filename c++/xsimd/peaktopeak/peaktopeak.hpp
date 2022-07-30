#include <vector>
#include <type_traits>

#include "minmax.hpp"

namespace peaktopeak {

long double peaktopeak(const std::vector<long double>& x)
{
    minmax::minmax_pair<long double> mm = minmax::minmax_scalar_loop(x);
    return mm.max - mm.min;
}

template<typename T>
using IsSignedIntegerType = std::enable_if_t<(std::is_signed<T>::value &&
                                              std::is_integral<T>::value)>;

template<typename T>
using IsNotSignedIntegerType = std::enable_if_t<(!(std::is_signed<T>::value &&
                                                   std::is_integral<T>::value))>;


template<typename T, typename = IsNotSignedIntegerType<T>>
T peaktopeak(const std::vector<T>& x)
{
    minmax::minmax_pair<T> mm = minmax::minmax(x);
    return mm.max - mm.min;
}


template<typename Int, typename = IsSignedIntegerType<Int>>
std::make_unsigned_t<Int>
peaktopeak(const std::vector<Int>& x)
{
    using Unsigned = std::make_unsigned_t<Int>;

    minmax::minmax_pair<Int> mm = minmax::minmax(x);
    if (mm.min < 0 && mm.max > 0) {
        return static_cast<Unsigned>(mm.max) + static_cast<Unsigned>(-mm.min);
    }
    else {
        return static_cast<Unsigned>(mm.max - mm.min);
    }
}

} // namespace peaktopeak
