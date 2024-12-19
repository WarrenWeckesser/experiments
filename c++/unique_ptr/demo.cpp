#include <cstdio>
#include <limits>
#include <memory>


extern "C"
double sum(size_t n, double *x)
{
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
}


double do_the_thing(size_t n)
{
    auto values = std::unique_ptr<double[]>{new (std::nothrow) double[n]};
    if (values.get() == nullptr) {
        fprintf(stderr, "memory allocation failed\n");
        return std::numeric_limits<double>::quiet_NaN();
    }
    for (size_t i = 0; i < n; ++i) {
        values[i] = i;
    }
    double total = sum(n, values.get());
    return total;
}


int main(int argc, char *argv[])
{
    double foo = do_the_thing(1000);
    printf("foo = %12.0f\n", foo);
}