

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "randtable.h"


void print_intarray(size_t ndim, size_t *dims, int *data)
{
    if (ndim == 1) {
        for (size_t i = 0; i < dims[0]; ++i) {
            printf("%4d ", data[i]);
        }
        printf("\n");
        return;
    }
    size_t outer_stride = 1;
    for (size_t i = 1; i < ndim; ++i) {
        outer_stride *= dims[i];
    }
    for (size_t i = 0; i < dims[0]; ++i) {
        print_intarray(ndim - 1, &dims[1], data + i*outer_stride);
        if (ndim > 2 && i < dims[0] - 1) {
            printf("\n");
        }
    }
}

//
// Simple (and incomplete) n-dimensional integer array.
//
typedef struct intarray {
    size_t ndim;
    size_t dims[32];
    int *data;
} intarray;


intarray *intarray_create(size_t ndim, size_t *dims)
{
    intarray *ia = calloc(1, sizeof(intarray));
    if (ia == NULL) {
        return NULL;
    }
    size_t size = 1;
    ia->ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        size_t dim = dims[i];
        ia->dims[i] = dim;
        size *= dim;
    }
    ia->data = calloc(size, sizeof(int));
    if (ia->data == NULL) {
        free(ia);
        return NULL;
    }
    return ia;
}

void intarray_destroy(intarray *ia)
{
    if (ia != NULL) {
        free(ia->data);
        free(ia);
    }
}

void intarray_fill(intarray *ia, int value)
{
    size_t size = 1;
    for (size_t i = 0; i < ia->ndim; ++i) {
        size *= ia->dims[i];
    }
    if (value == 0) {
        memset(ia->data, 0, size*sizeof(int));
    }
    else {
        for (size_t i = 0; i < size; ++i) {
            ia->data[i] = value;
        }
    }
}

void intarray_print(intarray *ia)
{
    print_intarray(ia->ndim, ia->dims, ia->data);
}


int main(int argc, char *argv[])
{
    int result;
    size_t n = 100;
    size_t ndims = 3;
    size_t dims[] = {2, 3, 4};
    size_t sums0[] = {60, 40};
    size_t sums1[] = {45, 25, 30};
    size_t sums2[] = {25, 25, 25, 25};
    size_t *counts[] = {sums0, sums1, sums2};

    srand(time(NULL));

    intarray *ia = intarray_create(ndims, dims);

    printf("Random contingency tables:\n");
    printf("-------------------------------------------------------\n");
    for (int k = 0; k < 3; ++k) {
        result = randtable(n, ndims, dims, counts, ia->data);
        if (result == -1) {
            fprintf(stderr, "randtable returned -1, indicating a memory allocation failure\n");
            exit(-1);
        }
        intarray_print(ia);
        printf("-------------------------------------------------------\n");
        intarray_fill(ia, 0);
    }

    return 0;
}
