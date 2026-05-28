
#include <iostream>
#include <chrono>
#include <complex>

using namespace std;

using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1> >;

#define N 200000

template<typename T>
double do_timing()
{
    std::complex<T> z[N];

    for (size_t k = 0; k < N; ++k) {
        z[k] = std::complex<T>((k+1)/(k+2.0), + k/(2.0*N));
    }

    std::complex<T> s{0.0, 0.0};
    double total_time = 0.0;

    for (int iter = 0; iter < 10; ++iter) {
        auto tstart = Clock::now();
        for (size_t k = 0; k < N; ++k) {
            auto w = log(z[k]);
            s += w;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        total_time += delta;
    }
    // Do something with s to prevent the entire loop from being
    // optimized away.
    cerr << "s = " << s << endl;
    return total_time;
}

int main()
{
    double t;
    cout << "N = " << N << endl;
    t = do_timing<float>();
    cout << "float:       " << t << endl;
    t = do_timing<double>();
    cout << "double:      " << t << endl;
    t = do_timing<long double>();
    cout << "long double: " << t << endl;
}
