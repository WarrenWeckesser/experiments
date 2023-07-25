
#include <iostream>
#include <iomanip>
#include "qd/dd_real.h"

using namespace std;

int main(int argc, char *argv[])
{
    unsigned oldcw;
    fpu_fix_start(&oldcw);

    double a = 0.1;
    double b = 0.2;
    double c = a + b;

    cout << "c  = " << setw(40) << setprecision(32) << scientific << c << endl;

    dd_real x0 = 0.1;
    dd_real y0 = 0.2;
    dd_real z0 = x0 + y0;

    cout << "z0 = " << setw(40) << setprecision(32) << z0 << endl;

    dd_real x1 = "0.1";
    dd_real y1 = "0.2";
    dd_real z1 = x1 + y1;

    cout << "z1 = " << setw(40) << setprecision(32) << z1 << endl;

    fpu_fix_end(&oldcw);
    return 0;
}
