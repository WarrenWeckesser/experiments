
#include <cmath>
#include <cstdio>
#include <iostream>
#include <boost/math/special_functions/owens_t.hpp>

using boost::math::owens_t;
using namespace std;


int main()
{
    vector<double> hvec = {0.0, 0.5, 1.0, 10.0};
    vector<double> avec = {0.0, 1.0, 100.0, INFINITY};
    printf("%8s  %8s  %21s\n", "h", "a", "t");
    for (auto h : hvec) {
        for (auto a : avec) {
            double t = owens_t(h, a);
            printf("%8.2f  %8.2f  %21.15e\n", h, a, t);
        }
    }
}
