#include <cmath>
#include <cstdio>

int main()
{
    double x = 1.25;
    double y = std::riemann_zeta(x);
    printf("y = %.17e\n", y);
}
