
#include <stdio.h>
#include <stdint.h>

void main()
{
    uint64_t x = 0b0000000000000000000000000000000111111111111111111111111111111111;
    int64_t i;
    // int8_t one = 1;
    for (i = 0; i < 35; ++i) {
        printf("%3ld  %12lu  %12ld\n", i, (1 << i), x & (1 << i));
    }
}