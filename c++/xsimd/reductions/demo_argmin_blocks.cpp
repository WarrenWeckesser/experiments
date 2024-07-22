#include <cstdio>
#include <vector>

#include "minmax.hpp"


int main()
{
    std::vector<short> v;

    constexpr std::size_t n = 323457;
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(static_cast<short>(i % 113));
    }
    int bad_count = 0;
    for (size_t k = 0; k < v.size(); ++k) {
        short tmp = v[k];
        v[k] = -99;
        minmax::value_index_pair<short> result = minmax::min_argmin(v);
        if (result.index != k) {
            printf("*** incorrect result: k = %zu  result.index = %zu\n", k, result.index);
            bad_count += 1;
            if (bad_count >= 5) {
                break;
            }
        }
        v[k] = tmp;
    }
    if (bad_count == 0) {
        printf("All tests passed.\n");
    }
}
