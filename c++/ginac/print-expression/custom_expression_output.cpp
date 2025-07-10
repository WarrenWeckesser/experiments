//
// This example is from the ginac tutorial.
// Some modifications have been necessary to get it to compile
// using ginac 1.8.9.
//

#include <iostream>
#include <ginac/ginac.h>

using namespace std;
using namespace GiNaC;


static void my_print(const ex & e)
{
    if (is_a<GiNaC::function>(e))
        cout << ex_to<GiNaC::function>(e).get_name();
    else
        cout << ex_to<basic>(e).class_name();
    cout << "(";
    size_t n = e.nops();
    if (n)
        for (size_t i=0; i<n; i++) {
            my_print(e.op(i));
            if (i != n-1)
                cout << ",";
        }
    else
        cout << e;
    cout << ")";
}

int main()
{
    symbol x("x"), y("y");
    my_print(pow(3, x) - 2 * sin(y / Pi));
    cout << endl;
    return 0;
}