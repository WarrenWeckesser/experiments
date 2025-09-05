#include <stdio.h>

int main(int argc, char *argv[])
{
    void *p;
    int x = 0;

    p = (x += 1, NULL);

    printf("x: %d\n", x);
    printf("p: %p\n", p);
    return 0;
}
