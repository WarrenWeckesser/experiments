#include <cln/cln.h>
#include <iostream>

int main(int argc, char *argv[])
{
    cln::cl_N z = cln::asin(2.0);
    std::cout << z << std::endl;

    return 0;
}
