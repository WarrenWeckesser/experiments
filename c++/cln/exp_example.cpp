
#include <iostream>
#include <cln/cln.h>

using namespace std;
using namespace cln;

int main(int argc, char *argv[])
{
    cl_F x = "-708.41357421875_40";
    cl_F y = exp(x);
    cout << y << endl;

    return 0;
}
