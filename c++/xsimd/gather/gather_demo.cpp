#include <cstdint>
#include <iostream>
#include "xsimd/xsimd.hpp"

using namespace std;

int main()
{
    float data[10] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f};
    xsimd::batch<int32_t> offsets = {0, 3, 1, 8};
    auto x = xsimd::batch<float>::gather(data, offsets);
    // This should print `x = (0.5, 3.5, 1.5, 8.5)`:
    cout << "x = " << x << endl;
}
