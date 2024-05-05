#include <iostream>
#include <cmath>


void show_isnormal(double x)
{
    std::cout << "isnormal(" << x << ") = " << std::isnormal(x) << std::endl;
}


int main(int argc, char *argv[])
{
    show_isnormal(5e-324);
    show_isnormal(1e-310);
    show_isnormal(1e-307);
    show_isnormal(1.0);

    return 0;
}