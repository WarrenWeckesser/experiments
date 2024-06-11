
//
//  ammprobcln.cpp
//
//  Calculation for a problem from the American Mathematical Monthly.
//  (C++ version that uses the CLN library)
//
//  Warren Weckesser, Dept. of Mathematics, Colgate University
//
//
//  Compile with:
//     g++ ammprobcln.cpp -o ammprobcln -lcln
//  or, if the library search options don't work,
//     g++ ammprobcln.cpp -o ammprobcln -I$CLNDIR/include $CLNDIR/lib/libcln.a
//

#include <iostream>
#include <cln/cln.h>

using namespace std;
using namespace cln;

int main(int argc, char **argv)
    {
    //
    // Declare the variables that we will use.  "cl_I" is the name of
    // the integer class created by the CLN library.
    //
    cl_I n, nlast, m;
    cl_I root;

    //
    // Check that two values were given in the command line.
    //
    if (argc != 3)
        {
        cout << "use: " << argv[0] << " first_integer last_integer\n";
        return(-1);
        }
    //
    // Assign the numbers given on the command line to n and nlast.
    //
    n     = argv[1];    // First command-line argument
    nlast = argv[2];    // Second command-line argument
    //
    // Check the integers from n to nlast.
    //
    while ( n <= nlast )
        {
        m = (n*n+1)*(n+1);
        if (sqrtp(m, &root))
            {
            cout << n << " " << m << " " << root << endl;
            }
        n = n + 1;
        }
    return(0);
    }
