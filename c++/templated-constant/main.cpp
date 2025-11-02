#include <cstdint>
#include <iostream>
#include "constants.h"

int main()
{
    std::cout << "foo<float>:   " << constants::foo<float> << std::endl;
    std::cout << "foo<double>:  " << constants::foo<double> << std::endl;
    std::cout << "foo<int>:     " << constants::foo<int> << std::endl;
    std::cout << "foo<long>:    " << constants::foo<long> << std::endl;

    // This line would fail to compile...
    // std::cout << "char:    " << constants::foo<char> << std::endl;

    std::cout << "fnv_init<uint32_t>:    " << constants::fnv_init<uint32_t> << std::endl;
    std::cout << "fnv_init<uint64_t>:    " << constants::fnv_init<uint64_t> << std::endl;

    return 0;
}