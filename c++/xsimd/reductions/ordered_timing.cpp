#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>

#include "ordered.hpp"

using namespace std;

using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1> >;


template<typename T>
double speedup_array(size_t n)
{
    double t1, t2;

    std::vector<T> x;
    x.resize(n, 2);
    x[n-2] = 10;
    x[n-1] = 0;

    auto tstart1 = Clock::now();
    auto result1 = ordered::is_increasing_v1(x);
    t1 = 1000*chrono::duration_cast<Second>(Clock::now() - tstart1).count();

    auto tstart2 = Clock::now();
    auto result2 = ordered::is_increasing_scalar_loop(x);
    t2 = 1000*chrono::duration_cast<Second>(Clock::now() - tstart2).count();

    if (result1 != result2) {
        cerr << "result1 != result2: " << result1 << " " << result2 << "\n";
    }

    return t2/t1;
}

template<typename T>
void check(const char *name, size_t n, int nrepeats)
{
    double s = 0.0;
    for (int k = 0; k < nrepeats; ++k) {
        s += speedup_array<T>(n);
    }
    cout << name << ":" << std::fixed << std::setprecision(2) << std::setw(6)
         << s/nrepeats << " (" << xsimd::simd_type<T>::size << ")\n";
}

int main()
{
    size_t n = 3499993;
    int nrepeats = 100;

    check<float>          ("float          ", n, nrepeats);
    check<double>         ("double         ", n, nrepeats);
    check<short>          ("short          ", n, nrepeats);
    check<unsigned short> ("unsigned short ", n, nrepeats);
    check<int>            ("int            ", n, nrepeats);
    check<long int>       ("long int       ", n, nrepeats);
    check<unsigned long>  ("unsigned long  ", n, nrepeats);
}
