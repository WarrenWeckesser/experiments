// compile with:
// $ gcc hello.c -licuuc -o hello
// or better:
// $ gcc hello.c $(icu-config --cflags --ldflags) -o hello

#include <stdio.h>
#include <unicode/uclean.h>

int main(int argc, char *argv[])
{
    UErrorCode status = U_ZERO_ERROR;

    u_init(&status);
    if (U_FAILURE(status)) {
        printf("Uh oh: %s\n", u_errorName(status));
        return 1;
    }
    printf("u_init success!\n");
    return 0;
}
