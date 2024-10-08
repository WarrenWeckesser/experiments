
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


//
// reflect_symm_index(j, m) maps an arbitrary integer j to the interval [0, m).
//
// * j can have any value.
// * m is assumed to be positive.
//
// The mapping reflects the indices about the edge of the base array.
//
// The mapping from j to [0, m) is via reflection about the edge of the array.
// That is, the "base" array is [0, 1, 2, ..., m-1].  To continue to the right,
// the indices count down: [m-1, m-2, ... 0], and then count up again [0, 1, ..., m-1].
// The same extension pattern is followed on the left.
//
// Example, with m = 5:
//                            ----extension--------|-----base----|----extension---- 
//                        j:  -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 10
// reflect_symm_index(j, 5):   3  4  4  3  2  1  0  0  1  2  3  4  4  3  2  1  0  0
//
static int64_t
reflect_symm_index(int64_t j, int64_t m)
{
    int64_t k = (j >= 0) ? (j % (2*m)) : (abs(j + 1) % (2*m));
    return (k >= m) ? (2*m - k - 1) : k;
}

//
// circular_wrap_index(j, m) maps an arbitrary integer j to the interval [0, m).
// The mapping makes the indices periodic.
//
// * j can have any value.
// * m is assumed to be positive.
//
// Example, with m = 5:
//                            ----extension--------|-----base----|----extension---- 
//                         j: -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 10
// circular_wrap_index(j, 5):  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0
//
static int64_t
circular_wrap_index(int64_t j, int64_t m)
{
    return (j >= 0) ? (j % m) : ((j % m + m) % m);
}


int main()
{
    int64_t m = 5;
    for (int64_t k = -7; k < 10; ++k) {
        printf("%3ld  %3ld  %3ld  %3ld\n", k, reflect_symm_index(k, m), k % m, circular_wrap_index(k, m));
    }
}
