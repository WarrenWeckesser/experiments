
#include <iostream>
#include <iomanip>
#include <cmath>
#include "qd/dd_real.h"

using namespace std;

int main(int argc, char *argv[])
{
    unsigned oldcw;
    fpu_fix_start(&oldcw);

    double x = -0.57113;
    double y = -0.90337;
    double t0 = x*(2.0 + x) + y*y;

    cout << "regular double: t0 = " << setw(25) << setprecision(17) << scientific << t0 << endl;

    dd_real xx = x;
    dd_real yy = y;
    dd_real t = xx*(2.0 + xx) + yy*yy;

    cout << "double double:  t1 = " << setw(25) << setprecision(17) << scientific << t.x[0] << endl;

    cout << "log1p(t0)/2 = " << setw(25) << setprecision(17) << scientific << 0.5*std::log1p(t0) << endl;

    double f1 = 0.5*std::log1p(t.x[0]);

    cout << "log1p(t1)/2 = " << setw(25) << setprecision(17) << scientific << f1 << endl;

    fpu_fix_end(&oldcw);
    return 0;
}
