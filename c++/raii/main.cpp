#include <cstdio>

class EnsureGIL
{
    long __save__;

public:

    EnsureGIL() {
        printf("EnsureGIL()\n");
        __save__ = 1;
    }

    ~EnsureGIL() {
        printf("~EnsureGIL()\n");
        __save__ = 0;
    }
};


int main()
{
    {
        EnsureGIL ensure_gil{};
        printf("AAAAA\n");
        goto skip_ahead;
        printf("BBBBB\n");
    }

skip_ahead:
    printf("Done.\n");

    return 0;
}