#include <Python.h>
#include "numpy/npy_math.h"
#include <limits>


int main()
{
    int nbits;

    nbits = std::numeric_limits<npy_float>::digits;
    printf("npy_float:      nbits = %d\n", nbits);

    nbits = std::numeric_limits<npy_double>::digits;
    printf("npy_double:     nbits = %d\n", nbits);

    nbits = std::numeric_limits<npy_longdouble>::digits;
    printf("npy_longdouble: nbits = %d\n", nbits);

    return 0;
}
