#include <cstdio>
#include <cmath>

int main(int argc, char *argv[])
{
    double x = std::pow(2.0, 54);
    double y = x;
    --y;
    printf("%.18e\n", x);
    printf("%.18e\n", y);
    if (x == y) {
        printf("x == y\n");
    }
    else {
        printf("x != y\n");
    }
}