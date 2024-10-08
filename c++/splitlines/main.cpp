#include <iostream>
#include <string>
#include <vector>

#include "splitlines.h"


using namespace std;

int main()
{
    string text = "apple\norange\ngrape";

    vector<string> lines = splitlines(text, "\n"s);

    for (auto &line: lines) {
        cout << "'" << line << "'\n";
    }
}