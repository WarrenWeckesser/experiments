#include "all_same.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int
main()
{
    std::vector<float> x{};
    for (int i = 0; i < 17; ++i) {
        bool x_all_same = all_same::all_same(x);
        if (!x_all_same) {
            std::cout << "Uh oh (1): x.size() = " << x.size()
                      << "  x_all_same  = " << x_all_same << std::endl;
        }
        x.push_back(1.5f);
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        x[i] = 3.0;
        bool x_all_same = all_same::all_same(x);
        if (x_all_same) {
            std::cout << "Uh oh (2): x.size() = " << x.size()
                      << "  x_all_same  = " << x_all_same << std::endl;
        }
        x[i] = v;
    }
    std::vector<double> y{0.5};
    for (int i = 0; i < 17; ++i) {
        y.push_back(1.5);
        bool y_all_same = all_same::all_same(y);
        if (y_all_same) {
            std::cout << "Uh oh (3): y.size() = " << y.size()
                      << "  y_all_same  = " << y_all_same << std::endl;
        }
        y.pop_back();
        y.push_back(0.5);
    }
    std::vector<int32_t> z{};
    for (int i = 0; i < 17; ++i) {
        bool z_all_same = all_same::all_same(z);
        if (!z_all_same) {
            std::cout << "Uh oh (4): z.size() = " << z.size()
                      << "  z_all_same  = " << z_all_same << std::endl;
        }
        z.push_back(10);
    }
}
