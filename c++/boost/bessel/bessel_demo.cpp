#include <cstdio>
#include <charconv>
#include <system_error>

#include <boost/math/special_functions/bessel.hpp>

template<typename T>
static inline void
print_value(const T x)
{
    char buf[100];
    const std::to_chars_result res = std::to_chars(buf, buf + sizeof(buf), x);
    if (res.ec == std::errc{}) {
        printf("%.*s", static_cast<int>(res.ptr - buf), buf);
    }
    else {
        printf("<to_chars() failed!>");
    }
}

using namespace boost::math;

int main(int argc, char *argv[])
{
    double nu = 0;
    double x = 11.791579157915791;
    double y = cyl_bessel_j(nu, x);
    print_value(x);
    printf("\n");
    print_value(y);
    printf("\n");
    return 0;
}
