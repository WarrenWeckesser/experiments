
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>


int randtable(size_t n, size_t ndims, size_t *dims, size_t **counts, int *out)
{
    // `out` is assumed to be contiguous.

    // Strides for the `out` array.  These count the number of ints; they
    // are not the number of bytes.
    size_t *element_strides = NULL;
    int status = 0;
    size_t **current_counts = calloc(ndims, sizeof(size_t *));
    if (current_counts == NULL) {
        return -1;
    }
    for (size_t i = 0; i < ndims; ++i) {
        current_counts[i] = calloc(dims[i], sizeof(size_t));
        if (current_counts[i] == NULL) {
            status = -1;
            goto finish;
        }
        memcpy(current_counts[i], counts[i], dims[i]*sizeof(size_t));
    }

    element_strides = calloc(ndims, sizeof(size_t));
    if (element_strides == NULL) {
        goto finish;
    }

    element_strides[ndims - 1] = 1;
    for (size_t i = ndims - 1; i > 0; --i) {
        element_strides[i - 1] = element_strides[i]*dims[i];
    }

    for (size_t i = n; i > 0; --i) {
        size_t out_index = 0;
        for (size_t j = 0; j < ndims; ++j) {
            size_t k = rand() % i;
            size_t m = 0;
            for (; m < dims[j]; ++m) {
                if (current_counts[j][m] > k) {
                    break;
                }
                k -= current_counts[j][m];
            }
            --current_counts[j][m];
            out_index += m*element_strides[j];
        }
        out[out_index] += 1;
    }

finish:
    free(element_strides);
    if (current_counts != NULL) {
        for (size_t i = 0; i < ndims; ++i) {
            free(current_counts[i]);
        }
        free(current_counts);
    }
    return status;
}
