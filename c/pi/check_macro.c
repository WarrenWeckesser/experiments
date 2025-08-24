
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
#if defined(M_PI)
    printf("M_PI is defined.\n");
#else
    printf("M_PI is not defined.\n");
#endif
    return 0;
}
