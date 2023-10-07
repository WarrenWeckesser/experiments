#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char *argv[])
{
    for (int k = 1; k < argc; ++k) {
        double x = stod(argv[k]);
        double r = remainder(x - 0.5, 1.0);
        cout << scientific << setw(25) << setprecision(17) << x;
        cout << scientific << setw(25) << setprecision(17) << r;
        if (fabs(r) < 1e-5) {
            cout << " (close to half-integer)";
        }
        cout << endl;
    }

    return 0;
}
