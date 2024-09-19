#include <iostream>
#include <string>
#include <vector>

#include "splitlines.h"


using namespace std;

int main()
{
    string text = R"(*
This is a test.
{@
Something here.
@}
)";

    vector<string> lines = splitlines(text, "\n"s);

    for (auto &line: lines) {
        cout << "'" << line << "'\n";
    }
}