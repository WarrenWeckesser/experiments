#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>

//
// Experiment with templated constants.
//

namespace constants {

template <typename T>
inline constexpr bool _false = false;

template <typename T>
struct _default {
  static_assert(
      _false<T>,
      "templated constant is not defined for this type.");
};

template <typename T>
inline constexpr T foo = _default<T>{};

template <> inline constexpr float foo<float> = 125.125;
template <> inline constexpr double foo<double> = 999.25;
template <> inline constexpr int foo<int> = -1;
template <> inline constexpr long foo<long> = -999;

template <typename T>
inline constexpr T fnv_init = _default<T>{};

template <> inline constexpr uint32_t fnv_init<uint32_t> = 0x811c9dc5;
template <> inline constexpr uint64_t fnv_init<uint64_t> = 0xcbf29ce484222325ULL;

}  // namespace constants

#endif  // CONSTANTS_H