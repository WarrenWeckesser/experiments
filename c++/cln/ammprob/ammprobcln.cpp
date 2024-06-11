//
//  ammprobcln.cpp
//
//  Calculation for a problem from the American Mathematical Monthly.
//  Search for solutions to the equation
//
//      n^3 + n^2 + n + 1 = x^2
//
//  This is a C++ version that uses the CLN library.
//
//
//  Compile with:
//     g++ ammprobcln.cpp -o ammprobcln -lcln
//

#include <iostream>
#include <cln/cln.h>

using namespace std;
using namespace cln;

int main(int argc, char **argv)
{
    cl_I n, nlast, m;
    cl_I root;

    if (argc != 3) {
        cout << "use: " << argv[0] << " first_integer last_integer\n";
        return(-1);
    }

    n     = argv[1];
    nlast = argv[2];

    while (n <= nlast) {
        m = (n*n + 1)*(n + 1);
        if (sqrtp(m, &root)) {
            cout << n << " " << m << " " << root << endl;
        }
        n = n + 1;
    }

    return(0);
}
