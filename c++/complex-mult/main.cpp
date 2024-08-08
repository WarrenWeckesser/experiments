
#include <cstdio>
#include <cstring>
#include <complex>
#include <charconv>
#include <system_error>


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

int main()
{
    std::complex x(-0.7173752049805099, 0.6966870282122177);
    std::complex y(25.441646575927734, -27.90408706665039);
    std::complex z = x*y;

    printf("z   = ");
    print_value(z.real());
    printf(" + ");
    print_value(z.imag());
    printf("i\n");

    double re1 = std::fma(x.real(), y.real(), -x.imag()*y.imag());
    printf("re1 = ");
    print_value(re1);
    printf("\n");

    double re2 = std::fma(x.imag(), -y.imag(), x.real()*y.real());
    printf("re2 = ");
    print_value(re2);
    printf("\n");

#ifdef __GNUC__
    printf("__GNUC__ is defined.\n");
#else
    printf("__GNUC__ is not defined.\n");
#endif
}