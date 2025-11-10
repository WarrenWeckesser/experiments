// See Section C.30 of "C++ Core Guidelines"
// https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md

#include <iostream>
#include "gsl_util.h"

int main()
{
    using std::cout;

    cout << "Before inner scope\n";

    {
        auto final_action = finally([] { cout << "Exiting inner scope\n"; });
        cout << "In inner scope\n";
        goto skip;
        cout << "This will not be printed\n";
    }

skip:
    cout << "Returning from main()\n";

    return 0;
}
