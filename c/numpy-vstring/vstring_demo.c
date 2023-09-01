#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vstring.h"
#include "vstring_print_utils.h"


// In this proof-of-concept demo, UTF-8 is not used.  The strings
// are just sequences of bytes; the encoding could be considered
// to be ASCII or Latin-1.

#define ARRAY_LEN 8

int main(int argc, char *argv[])
{
    if (sizeof(char *) != sizeof(size_t)) {
        fprintf(stderr, "error: sizeof(char *) != sizeof(size_t)");
        exit(-1);
    }

    int dummy = 1;
    if ((*(char *) &dummy) != 1) {
        fprintf(stderr, "this proof-of-concept code is designed to run on a little-endian system");
        exit(-1);
    }

    int n = argc - 1;
    npy_static_string *arr = calloc(n, sizeof(npy_static_string));
    if (arr == NULL) {
        fprintf(stderr, "calloc() failed.\n");
        exit(-1);
    }

    for (int i = 0; i < n; ++i) {
        size_t size = strlen(argv[i + 1]);
        if (size == 1 && argv[i + 1][0] == '?') {
            // In this demo program, if the argument on the command line
            // is "?", we create a not-a-string element.  In the actual
            // implementation in NumPy, presumably there would be an object
            // called `np.notastring` (or `np.nas`, `np.not_a_string`,
            // `np.nastring`?), analogous to `np.nan` for floats and `np.nat`
            // for datetimes and timedeltas.
            assign_notastring(arr + i);
        }
        else {
            assign_string(arr + i, size, argv[i + 1]);
        }
    }

#ifdef MOCK_BE
    printf("MOCK_BE is enabled.\n");
#endif

    printf("n = %d\n", n);

    printf("raw dump:\n");
    for (size_t i = 0; i < n; ++i) {
        raw_dump(arr + i);
    }

    printf("vstring array:\n");
    print_vstring_array(n, arr);

    free(arr);
    return 0;
}
