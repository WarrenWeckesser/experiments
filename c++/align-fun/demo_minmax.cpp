#include <iostream>
#include <cstdint>
#include <vector>

#include "minmax.h"


int main()
{
    std::vector<float> x{0.0, 1.5, -3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                         3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                         2.5, 1.0, 0.0, 1.0, -1.5, 13.0, 0.0};
    // Force a smaller alignment by passing &x[1]
    auto x_mm = minmax::minmax_try_aligned(x.size()-1, &x[1]);
    std::cout << "x_mm = " << x_mm << std::endl;

    double y[] = {0.5, 1.5, 3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                  3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                  2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.0};
    auto y_mm = minmax::minmax_try_aligned(std::size(y), y);
    std::cout << "y_mm = " << y_mm << std::endl;

    std::vector<uint16_t> m{145,
                            2, 3, 4, 10, 100, 10, 19,
                            90, 99, 90, 22, 234, 132, 18, 447,
                            212, 899, 500, 618, 1000, 233, 661, 190,
                            133, 312, 498, 2513, 811, 1};
    auto m_mm = minmax::minmax_try_aligned(m.size(), &m[0]);
    std::cout << "m_mm = " << m_mm << std::endl;

    std::vector<int8_t> b{34, -100, 125, -10, -125, 17, -1, 22, 121,
                          18, -89, -90, 101, 60, 34, 34, 34, 34, 99,
                          111, -23, 81, 48, -4, 77, 33, 12, 13, -90,
                          -10, -68, 50, 50, 50, 50, 50, 50, 50, 50,
                          50, 50, 50, 50, 50, 50, 45, 0, 0, 9, 9,
                          1, 2, 3, 4, 5, 6, 7, 8, 126, 0};
    auto b_mm = minmax::minmax_try_aligned(b.size(), &b[0]);
    std::cout << "b_mm = " << b_mm << std::endl;
}
