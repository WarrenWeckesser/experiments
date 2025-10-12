#include <iostream>
#include <cstdint>
#include <vector>

#include "peaktopeak.hpp"


int main()
{
    std::vector<float> x{0.5, 1.5, 3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                         3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                         2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.0};
    float xptp = peaktopeak::peaktopeak(x);
    std::cout << "peaktopeak(x) = " << xptp << std::endl;

    std::vector<double> y{0.5, 1.5, 3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                          3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                          2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.0};
    auto yptp = peaktopeak::peaktopeak(y);
    std::cout << "peaktopeak(y) = " << yptp << std::endl;

    std::vector<long double> z{0.5, 1.5, 3.0, 0.5, 4.5, 0.1, 0.2, 0.2,
                               3.5, 1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
                               2.5, 1.0, 0.0, 1.0, -1.5, 3.0, 0.0};
    auto zptp = peaktopeak::peaktopeak(z);
    std::cout << "peaktopeak(z) = " << zptp << std::endl;

    std::vector<uint16_t> m{34, 999, 1, 23, 100, 100, 100, 100, 20,
                            89, 90, 22, 234, 132, 18, 447, 212, 899,
                            500, 618, 1000, 233, 661, 124, 398, 312,
                            498, 253, 811, 498};
    auto mptp = peaktopeak::peaktopeak(m);
    std::cout << "peaktopeak(m) = " << mptp << std::endl;

    std::vector<int16_t> a{1034, 32000, 1, 23, 100, 100, 21000, 100, 20,
                           9, 9, 22, -5000, 132, 18, 1447, 212, 899,
                           400, 7618, 1000, 33, 9661, -1024, 98, 12};
    auto aptp = peaktopeak::peaktopeak(a);
    std::cout << "peaktopeak(a) = " << aptp << std::endl;

    std::vector<int8_t> b{34, -100, 125, -10, -125, 17, -1, 22, 121,
                          18, -89, -90, 101, 60, 34, 34, 34, 34, 99,
                          111, -23, 81, 48, -4, 77, 33, 12, 13, -90,
                          -10, -68, 50, 50, 50, 50, 50, 50, 50, 50,
                          50, 50, 50, 50, 50, 50, 45, 0, 0, 9, 9,
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    auto bptp = peaktopeak::peaktopeak(b);
    std::cout << "peaktopeak(b) = " << +bptp << std::endl;
}
