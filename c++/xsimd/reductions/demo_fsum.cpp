#include <iostream>
#include <iomanip>
#include <vector>
#include "fsum.hpp"

using namespace std;

int main()
{
    vector<double> x{1.0, 2.0, 0.25, -2.0, -2e-8,
                     10.0, -3.0, 2.0, -10.0,
                     3.0, 0.0, 0.0, 0.0, 1e-8,
                     -1.0, 1.5, 1e-8, -0.75};

    double s1 = fsum::fsum_scalar_loop(x);

    cout << "s1 = " << scientific << setprecision(20) << s1 << endl;

    double s2 = fsum::fsum(x);

    cout << "s2 = " << scientific << setprecision(20) << s2 << endl;

    return 0;
}
