#include <cstdio>
#include <charconv>
#include <string>
#include <system_error>

#include <boost/math/special_functions/bessel.hpp>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


typedef boost::math::policies::policy<
    boost::math::policies::promote_float<false>,
    boost::math::policies::promote_double<false>
> NoPromotionPolicy;

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
    if (argc != 3 ) {
        printf("use: %s nu x\n", argv[0]);
        return -1;
    }

#ifdef BOOST_MATH_GIT
    printf("boost/math git: " STR(BOOST_MATH_GIT) "\n");
#endif

    double nu = std::stod(argv[1]);
    double x = std::stod(argv[2]);
    double y = cyl_bessel_i(nu, x, NoPromotionPolicy());
    // double y = cyl_bessel_i(nu, x);
    print_value(nu);
    printf("\n");
    print_value(x);
    printf("\n");
    print_value(y);
    printf("\n");
    return 0;
}
