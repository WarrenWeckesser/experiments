#include <cstdio>
#include <limits>
#include <memory>


void do_the_thing(size_t n)
{
    double init = 0.0;
    // Allocate an array and wrap it in a unique_ptr.
    auto values = std::unique_ptr<double[]>{new (std::nothrow) double[n]()};
    if (values.get() == nullptr) {
        fprintf(stderr, "memory allocation failed\n");
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        if (values[i] != init) {
            printf("%25.15e\n", values[i]);
        }
        values[i] = 12.5;
    }
}

// double do_the_thing2(size_t n)
// {
//     // Allocate an array initialized to 0 and wrap it in a unique_ptr.
//     auto values = std::unique_ptr<double[]>{new (std::nothrow) double[n]()};
//     if (values.get() == nullptr) {
//         fprintf(stderr, "memory allocation failed\n");
//         return std::numeric_limits<double>::quiet_NaN();
//     }
//     for (size_t i = 1; i < n; ++i) {
//         values[i] = i;
//     }
//     double total = sum(n, values.get());
//     return total;
// }

int main(int argc, char *argv[])
{
    do_the_thing(5000000);
    do_the_thing(5000000);
}
