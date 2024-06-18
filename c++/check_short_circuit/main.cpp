
#include <cstdio>


double foo(double x, double y)
{
    return x*x + y*y;
}

int main()
{
    double x = 1e200;
    double y = 1e200;

    if (x < 2 && x > -2 && y < 2 && y > -2 && foo(x, y) < 1) {
        printf("Here I am!\n");
    }
    else {
        printf("Condition failed!\n");
    }
}