
#include <cstdio>
#include <cstdlib>
#include "strided_vector.h"


double *alloc_data(size_t n)
{
    double *data = (double *) calloc(n, sizeof(double));
    double s = 1.0;
    for (size_t k = 0; k < n; ++k) {
        data[k] = s*(k + 1.0);
        s *= -1.0;
    }
    return data;
}

int main(void)
{
    size_t n = 10;
    double *data = alloc_data(n);

    // auto v = StridedVector<double>(data, (n+2)/3, 3*sizeof(data[0]));
    auto v = make_strided_vector(data, (n+2)/3, 3*sizeof(data[0]));

    printf("len    = %ld\n", v.size());
    printf("stride = %ld\n", v.get_stride());

    printf("strided vector v:\n");
    for (int k =0; k < v.size(); ++k) {
        printf("v[%d] = %f\n", k, v[k]);
    }

    v[1] = -1.5;
    printf("after assigning v[1] = -1.5:\n");
    for (int k =0; k < v.size(); ++k) {
        printf("v[%d] = %f\n", k, v[k]);
    }

    free(data);

    return 0;
}