#ifndef PRINT_STUFF_H
#define PRINT_STUFF_H

#include <cstdio>
#include <charconv>
#include <system_error>

#include "log1p_complex.h"


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
static void
print_doubled_t(const log1p_complex::doubled_t<T>& x)
{
    print_value(x.upper);
    printf("\n");
    print_value(x.lower);
    printf("\n");
}

#endif
