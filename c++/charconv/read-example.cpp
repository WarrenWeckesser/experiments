#include <cstdio>
#include <charconv>
#include <cstring>
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

template<typename T>
T parse(const char *start, const char *end)
{
    T value;
    const std::from_chars_result result = std::from_chars(start, end, value);
    // XXX Should check result for errors!
    return value;
}

using namespace std;

int main()
{
    const char text[] = "1.234567890123456789012345678901234567890";

    float x = parse<float>(text, text + strlen(text));
    printf("x = ");
    print_value(x);
    printf("\n");

    double y = parse<double>(text, text + strlen(text));
    printf("y = ");
    print_value(y);
    printf("\n");

    long double z = parse<long double>(text, text + strlen(text));
    printf("z = ");
    print_value(z);
    printf("\n");
}
