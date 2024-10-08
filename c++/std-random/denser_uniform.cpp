
#include <cstdio>
#include <random>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <cstring>
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


// XXX __builtin_clz and __builtin_clzl are GCC only!
// For MSVC, use _ReverseScan and _ReverseScan64.
// For other compilers, find the equivalent functions or implement
// from scratch.

int clz(uint32_t x)
{
    return __builtin_clz(x);
}

int clz(uint64_t x)
{
    return __builtin_clzl(x);
}


template<int nbits>
int random_exponent(std::mt19937_64 &gen)
{
    static_assert(nbits == 32 || nbits == 64, "nbits must be 32 or 64");
    int p;
    if (nbits == 32) {
        std::uniform_int_distribution<uint32_t> random_uint32(0);
        uint32_t r = random_uint32(gen);
        if (r == 0) {
            p = 32;
        }
        else {
            p = clz(r);
        }
    }
    else if (nbits == 64) {
        std::uniform_int_distribution<uint64_t> random_uint64(0);
        uint64_t r = random_uint64(gen);
        if (r == 0) {
            p = 64;
        }
        else {
            p = clz(r);
        }
    }
    return p;
}

//
// denser_standard_uniform generates samples from the
// uniform distribution on [0, 1).
// It is guaranteed that 1.0 will not be returned.
// The code uses an extra random unsigned integer (size
// determined by the nbits template parameter) to allow,
// for example, *all* of the possible floating point values
// in the intervals [1/4, 1/2), [1/8, 1/4), etc. down to an
// interval size determined by nbits, to be possible outputs.
// The number of intervals for which this holds is nbits+1
// (XXX double-check that!).
//
// XXX nbits needs a more descriptive name.
//
// The code depends on IEEE floating point representation
// of 32 bit single precision and 64 bit double precision
// values.  It does bit twiddling to construct a floating point
// number from integers. (XXX what about endianess? Check this.)
//
// With a bit more templating cleverness, the following two
// functions could probably be implemented as one templated
// function.
//

template<int nbits = 32>
void denser_standard_uniform(std::mt19937_64 &gen, double &u)
{
    assert(sizeof(unsigned long) == sizeof(uint64_t));
    assert(sizeof(double) == sizeof(uint64_t));

    int p = random_exponent<nbits>(gen);
    std::uniform_int_distribution<uint64_t> random_uint64(0);
    uint64_t m = random_uint64(gen) >> 12;
    // Assemble m and p into a double.
    m |= static_cast<uint64_t>(1022 - p) << 52;
    memcpy(reinterpret_cast<char *>(&u),
           reinterpret_cast<const char*>(&m),
           sizeof(m));
}

template<int nbits = 32>
void denser_standard_uniform(std::mt19937_64 &gen, float &u)
{
    assert(sizeof(unsigned) == sizeof(uint32_t));
    assert(sizeof(float) == sizeof(uint32_t));

    int p = random_exponent<nbits>(gen);
    std::uniform_int_distribution<uint32_t> random_uint32(0);
    uint32_t m = random_uint32(gen) >> 9;
    // Assemble m and p into a float.
    m |= static_cast<uint32_t>(126 - p) << 23;
    memcpy(reinterpret_cast<char*>(&u),
           reinterpret_cast<const char*>(&m),
           sizeof(m));
}


int main(int argc, char *argv[])
{
    std::random_device rd;
    std::mt19937_64 gen(rd());

    int n;
    char type_code;
    if (argc != 3) {
        printf("use: denser_uniform type_code n\n");
        printf("type_code is either f or d (for float or double.\n");
        printf("n is the number of samples to generate.\n");
        return 0;
    }
    type_code = argv[1][0];
    if (type_code != 'd' && type_code != 'f') {
        printf("type_code must be 'f' or 'd'.\n");
        return -1;
    }
    n = atoi(argv[2]);

    for (int k = 0; k < n; ++k) {
        if (type_code == 'd') {
            double u;
            denser_standard_uniform(gen, u);
            print_value(u);
            printf("\n");
        }
        else {
            float u;
            denser_standard_uniform<64>(gen, u);
            print_value(u);
            printf("\n");
        }
    }
}
