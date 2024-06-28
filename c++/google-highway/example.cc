// Blindly experimenting with google highway.  This file is barely
// a step or two up the steep learning curve.
//
// This is a drastically pared down and modified version of
// 'hwy/examples/benchmark.cc' from the Google highway repository.
// The license of 'benchmark.cc' is:
//
// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "example.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"

HWY_BEFORE_NAMESPACE();
namespace stuff {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template<typename T>
void square() {
  const T in[16] = {1, 2, 3, 4, 5, 6};
  T out[16];
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < 16; i += hn::Lanes(d)) {
    const auto vec = hn::LoadU(d, in + i);
    auto result = hn::Mul(vec, vec);
    result = hn::Add(result, result);
    hn::StoreU(result, d, out + i);
  }
  printf("in[2] = %.0Lf, out[2] = %.1Lf\n",
         static_cast<long double>(in[2]),
         static_cast<long double>(out[2]));
}

}  // namespace HWY_NAMESPACE
}  // namespace stuff
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

int main(int /*argc*/, char** /*argv*/) {
  stuff::HWY_NAMESPACE::square<float>();
  stuff::HWY_NAMESPACE::square<double>();
  return 0;
}

#endif  // HWY_ONCE
