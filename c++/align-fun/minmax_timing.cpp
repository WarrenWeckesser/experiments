#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>

#include "minmax.h"

using namespace std;

using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1> >;

int main()
{
    const int n = 8000004;
    std::vector<double> x;
    x.resize(n);
    std::vector<double> y;
    y.resize(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = 0.75*sin(700.1*i/n);
        y[i] = x[i];
    }

    {
    auto tstart = Clock::now();
    auto x_mm = minmax::minmax(std::size(x)-3, &x[3]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "x_mm = " << x_mm << endl;
    cout << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto y_mm = minmax::minmax2(std::size(y)-3, &y[3]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "y_mm = " << y_mm << endl;
    cout << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto x_mm = minmax::minmax_scalar_loop(std::size(x)-3, &x[3]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "x_mm = " << x_mm << endl;
    cout << delta << " ms" << endl;
    }

    const int m = 1000000;
    std::vector<int16_t> xi;
    xi.resize(m);
    std::vector<int16_t> yi;
    yi.resize(m);
    std::vector<int16_t> zi;
    zi.resize(m);
    for (size_t i = 0; i < m; ++i) {
        xi[i] = 1000*sin(700.1*i/n) + 10;
        yi[i] = xi[i];
        zi[i] = xi[i];
    }

    {
    auto tstart = Clock::now();
    auto xi_mm = minmax::minmax(std::size(xi)-1, &xi[1]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "xi_mm = " << xi_mm << endl;
    cout << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto yi_mm = minmax::minmax2(std::size(yi)-1, &yi[1]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "yi_mm = " << yi_mm << endl;
    cout << delta << " ms"  << endl;
    }

    {
    auto tstart = Clock::now();
    auto zi_mm = minmax::minmax_scalar_loop(std::size(zi)-1, &zi[1]);
    auto delta = 1000*chrono::duration_cast<Second>(Clock::now() - tstart).count();
    cout << "zi_mm = " << zi_mm << endl;
    cout << delta << " ms" << endl;
    }
}
