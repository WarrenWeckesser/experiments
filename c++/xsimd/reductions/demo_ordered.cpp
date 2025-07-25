#include <iostream>
#include <cstdint>
#include <vector>

#include "ordered.hpp"


int main()
{
    std::vector<long> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10};
    auto incr1 = ordered::is_increasing_v1(x);
    std::cout << "x is increasing? v1: " << incr1 << std::endl;
    auto incr2 = ordered::is_increasing_v2(x);
    std::cout << "x is increasing? v2: " << incr2 << std::endl;
}
