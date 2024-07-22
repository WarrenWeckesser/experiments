#include <cstdio>
//#include <cstdint>
#include <vector>

#include "minmax.hpp"


int main()
{
    std::vector<double> x{-2.0, 1.5, 3.0, 5.5, 4.5, 0.1, 0.2, 0.2,
                          3.5, -2.5, 2.5, 2.5, 2.5, 2.5, 0.25, 2.5,
                          2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.3, -0.25,
                          0.1, 4.5};
    for (size_t k = 0; k < x.size(); ++k) {
        float tmp = x[k];
        x[k] = -99.0;
        minmax::value_index_pair<double> result = minmax::min_argmin(x);
        printf("k = %4ld   kmin = %4ld\n", k, result.index);
        x[k] = tmp;
    }
    printf("\n");

    std::vector<short> v{-2, 1, 3, 5, 4, 0, 0, 0,
                         3, -2, 2, 2, 2, 2, 0, 2,
                         2, 1, 0, 1, -1, -3, 0};
    for (size_t k = 0; k < v.size(); ++k) {
        float tmp = v[k];
        v[k] = -99;
        minmax::value_index_pair<short> result = minmax::min_argmin(v);
        printf("k = %4ld   kmin = %4ld\n", k, result.index);
        v[k] = tmp;
    }
}
