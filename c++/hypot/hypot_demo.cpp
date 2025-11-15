#include <cmath>
#include <iostream>

int
main(int argc, char *argv[])
{
    double x = 1.0;
    double y = 2.0;
    double z = 3.0;
    double h = std::hypot(x, y, z);
    std::cout << "h = " << h << std::endl;

    double z2 = INFINITY;
    double h2 = std::hypot(x, y, z2);
    std::cout << "h2 = " << h2 << std::endl;
}
