#include <string>
#include <iostream>
#include <vector>
#include <variant>

using namespace std;

typedef variant<string,
                vector<string>,
                vector<vector<string>>
                > mixed_t;

int main()
{
    vector<string> things {"shrimp", "spaghetti", "peas"};
    vector<string> objects {"chair", "pencil", "wagon", "hat"};
    vector<mixed_t> x{"plate", "of", things, "more", objects};

    for (auto &item : x) {
        cout << item.index() << endl;
        if (holds_alternative<string>(item)) {
            string value = std::get<string>(item);
            cout << "string: " << value << endl;
        }
        else if (holds_alternative<vector<string>>(item)) {
            vector<string> value = std::get<vector<string>>(item);
            cout << "vector:";
            for (auto &vector_item : value) {
                cout << " " << vector_item;
            }
            cout << endl;
        }
        else if (holds_alternative<vector<vector<string>>>(item)) {
            cout << "vector of vectors" << endl;
        }
        else {
            cout << "Unkown variant type! (This should not happen.)" << endl;
        }
    }
}
