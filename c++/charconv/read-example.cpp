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
    return value;
}

using namespace std;

int main()
{
    const char text[] = "1.2345";
    double x = parse<double>(text, text + strlen(text));
    printf("x = ");
    print_value(x);
    printf("\n");
}
