// Requires C++11 or later (for *this* code, not for Boost in general).

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/special_functions/beta.hpp>

using namespace boost::math;
using namespace std;

struct params {
    int k;
    int n;
    double p;
};

int main(int argc, char *argv[])
{
    vector<params> paramset = {
        {     300,      1000, 0.3},
        {    3000,     10000, 0.3},
        {   30000,    100000, 0.3},
        {  300000,   1000000, 0.3},
        { 3000000,  10000000, 0.3},
        {30000000, 100000000, 0.3},
        {30010000, 100000000, 0.3},
        {29990000, 100000000, 0.3},
        {29950000, 100000000, 0.3}
    };

    cout << setw(10) << "k";
    cout << setw(10) << "n";
    cout << setw(5) << "p";
    cout << setw(23) << "binomial.cdf(k, n, p)";
    cout << endl;

    for (auto [k, n, p] : paramset) {
        double value = ibeta(n - k, k + 1, 1 - p);
        cout << setw(10) << k;
        cout << setw(10) << n;
        cout << fixed << setw(5) << setprecision(2) << p;
        cout << scientific << setw(23) << setprecision(12) << value;
        cout << endl;
    }

    return 0;
}
