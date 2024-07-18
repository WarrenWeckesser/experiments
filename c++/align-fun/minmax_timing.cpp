#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>

#include "minmax.h"

using namespace std;

using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1> >;


template<typename T>
void test_float_point_array(size_t offset)
{
    const size_t n = 500000;
    std::vector<T> x;
    x.resize(n);
    std::vector<T> y;
    y.resize(n);
    std::vector<T> z;
    z.resize(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = 0.75*sin(700.1*i/n) + 0.25;
        y[i] = x[i];
        z[i] = x[i];
    }
    x[n/2] = 2.0;
    y[n/2] = 2.0;
    z[n/2] = 2.0;

    {
    auto tstart = Clock::now();
    auto x_mm = minmax::minmax_unaligned(std::size(x)-offset, &x[offset]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "minmax_unaligned:   x_mm = " << x_mm;
    cout << "   " << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto y_mm = minmax::minmax_try_aligned(std::size(y)-offset, &y[offset]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "minmax_try_aligned: y_mm = " << y_mm;
    cout << "   " << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto z_mm = minmax::minmax_scalar_loop(std::size(z)-offset, &z[offset]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "minmax_scalar_loop: z_mm = " << z_mm;
    cout << "   " << delta << " ms" << endl;
    }
}

template<typename T>
void test_integer_array(size_t offset)
{
    const int m = 8000000;
    std::vector<T> xi;
    xi.resize(m);
    std::vector<T> yi;
    yi.resize(m);
    std::vector<T> zi;
    zi.resize(m);
    for (size_t i = 0; i < m; ++i) {
        xi[i] = 1000*sin(700.1*i/m) + 10;
        yi[i] = xi[i];
        zi[i] = xi[i];
    }

    {
    auto tstart = Clock::now();
    auto xi_mm = minmax::minmax_unaligned(std::size(xi)-offset, &xi[offset]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "xi_mm = " << xi_mm;
    cout << "   " << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto yi_mm = minmax::minmax_try_aligned(std::size(yi)-offset, &yi[offset]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "yi_mm = " << yi_mm;
    cout << "   " << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto zi_mm = minmax::minmax_scalar_loop(std::size(zi)-offset, &zi[offset]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "zi_mm = " << zi_mm;
    cout << "   " << delta << " ms" << endl;
    }
}

int main()
{
    cout << "float, offset=0:\n";
    test_float_point_array<float>(0);
    cout << "float, offset=1:\n";
    test_float_point_array<float>(1);


    cout << "double, offset=0:\n";
    test_float_point_array<double>(0);
    cout << "double, offset=1:\n";
    test_float_point_array<double>(1);
    cout << endl;

/*
    cout << "int16_t, offset=0:\n";
    test_integer_array<int16_t>(0);
    cout << "int32_t, offset=0:\n";
    test_integer_array<int32_t>(0);
    cout << "int64_t, offset=0:\n";
    test_integer_array<int64_t>(0);
*/
}
