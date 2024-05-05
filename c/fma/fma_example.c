#include <stdio.h>
#include <math.h>
#include <pmmintrin.h>

#define N 100000000

double x[N];
double y[N];


void f1(int n, double *u, double x0, double a, double *y)
{
    int i;

    printf("f1: n = %d\n", n);
    for (i = 0; i < n; ++i) {
        x0 = a * x0 + *u;
        *y = x0;
        ++u;
        ++y;
    }
    printf("f1: done\n");
}

void f2(int n, double *u, double x0, double a, double *y)
{
    int i;
    double q[2] = {1.0, -1.0};
    __m128d d = _mm_load_pd(q);

    printf("f2: n = %d\n", n);
    for (i = 0; i < n; ++i) {
        x0 = a * x0 + *u;
        *y = x0;
        ++u;
        ++y;
    }
    printf("f2: done\n");
}

void printvec(int n, double *v)
{
    int i;

    for (i = 0; i < n; ++i) {
        printf("%9.5f\n", v[i]);
    }
}

int main(int argc, char *argv[])
{
    double a = 0.5;
    double x0 = 2.0;
    int i;

    for (i = 0; i < N; ++i) {
        x[i] = sqrt(1.0*i);
    }
    f1(N, x, x0, a, y);
    return 0;
}
