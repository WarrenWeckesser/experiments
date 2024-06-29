
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <cstddef>


void foo(size_t n, short *x)
{
    void *ptr = x;
    size_t space_before = n*sizeof(short);
    printf("ptr:   %p\n", ptr);
    printf("space_before: %ld\n", space_before);
    size_t space_after = space_before;
    void *p = std::align(8, 8, ptr, space_after);
    if (p) {
        printf("ptr:   %p\n", ptr);
        printf("space_after:  %ld\n", space_after);
        size_t delta = space_before - space_after;
        printf("delta: %ld\n", delta);
        printf("number of skipped elements before alignment is %ld\n", delta / sizeof(short));
    }
    else {
        printf("align failed!\n");
    }
}

#define N 25

int main()
{
    short x[N];
    // Force a smaller alignment for this test...
    foo(N - 1, &x[1]);
}
