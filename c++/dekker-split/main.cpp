#include <cstdio>
#include <cmath>
#include <limits>
#include <charconv>
#include <system_error>


using namespace std;


template <typename T>
class doubled_t {
public:
    T upper;
    T lower;
};

//
// Dekker splitting.  See, for example, Theorem 1 of
//
//   Seppa Linnainmaa, Software for Double-Precision Floating-Point
//   Computations, ACM Transactions on Mathematical Software, Vol 7, No 3,
//   September 1981, pages 272-283.
//
template <typename T>
static void
split(T x, doubled_t<T>& out)
{
    constexpr int halfprec = (std::numeric_limits<T>::digits + 1)/2;
    T t = ((1ul << halfprec) + 1)*x;
    // The compiler must not be allowed to optimize away this expression:
    out.upper = t - (t - x);
    out.lower = x - out.upper;
}


template<typename T>
void print_value(const T x)
{
    char buf[100];
    const to_chars_result res = to_chars(buf, buf + sizeof(buf), x);
    if (res.ec == errc{}) {
        printf("%.*s", static_cast<int>(res.ptr - buf), buf);
    }
    else {
        printf("<to_chars() failed!>");
    }
}

template<typename T>
void print_double_t(const doubled_t<T>& x)
{
    print_value(x.upper);
    printf("\n");
    print_value(x.lower);
    printf("\n");
}

template<typename T>
void demo(T x)
{
    doubled_t<T> out;

    printf("x:\n");
    print_value(x);
    printf("\n");

    split(x, out);
    printf("split(x):\n");
    print_double_t(out);
}

int main()
{
    printf("Split a double...\n");
    double x = 1.0/7;
    demo(x);
    printf("\n");

    printf("Split a long double...\n");
    long double y = 1.0L/3;
    demo(y);

    return 0;
}
