#include <cmath>
#include <iostream>

int
main(int argc, char *argv[])
{
    double x = 1.0;
    double y = 2.0;
    double z = 3.0;

    double hxy = std::hypot(x, y);
    std::cout << "hxy  = " << hxy << std::endl;

    double hxyz = std::hypot(x, y, z);
    std::cout << "hxyz = " << hxyz << std::endl;

    double w = INFINITY;
    double hxw = std::hypot(x, w);
    std::cout << "hxy  = " << hxw << std::endl;

    double hxwz = std::hypot(x, w, z);
    std::cout << "hxwz = " << hxwz << "  (This should be inf, but there is a bug in gcc's implementation of std::hypot" << std::endl;
}
