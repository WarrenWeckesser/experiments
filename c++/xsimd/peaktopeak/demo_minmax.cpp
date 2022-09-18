#include <iostream>
#include <cstdint>
#include <vector>

#include "minmax.hpp"


int main()
{
    std::vector<float> x{0.5, 1.5, 3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                         3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                         2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.0};
    auto x_mm = minmax::minmax(x);
    std::cout << "x_mm = (" << x_mm.min << ", " << x_mm.max << ')' << std::endl;


    std::vector<double> y{0.5, 1.5, 3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                          3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                          2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.0};
    auto y_mm = minmax::minmax(y);
    std::cout << "y_mm = (" << y_mm.min << ", " << y_mm.max << ')' << std::endl;

    std::vector<uint16_t> m{34, 999, 1, 23, 100, 100, 100, 100, 20,
                            89, 90, 22, 234, 132, 18, 447, 212, 899,
                            500, 618, 1000, 233, 661, 124, 398, 312,
                            498, 253, 811, 498};
    auto m_mm = minmax::minmax(m);
    std::cout << "m_mm = (" << m_mm.min << ", " << m_mm.max << ')' << std::endl;

    std::vector<int8_t> b{34, -100, 125, -10, -125, 17, -1, 22, 121,
                          18, -89, -90, 101, 60, 34, 34, 34, 34, 99,
                          111, -23, 81, 48, -4, 77, 33, 12, 13, -90,
                          -10, -68, 50, 50, 50, 50, 50, 50, 50, 50,
                          50, 50, 50, 50, 50, 50, 45, 0, 0, 9, 9,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    auto b_mm = minmax::minmax(b);
    std::cout << "b_mm = (" << +b_mm.min << ", " << +b_mm.max << ')' << std::endl;
}
