
#include <stdio.h>

#if defined(__APPLE__)
#define FOO "APPLE!\n"
#else
#define FOO "Something else\n"
#endif

int main(int argc, char *argv[])
{
    printf(FOO);

    return 0;
}
