#include <iostream>

#include <set>

int main ()
{
    std::set<std::string> s;
    std::cout << "Adding 'Hello' and 'World' to the set twice" << std::endl;
  
    s.insert("Hello");
    s.insert("World");
    s.insert("Hello");
    s.insert("World");

    std::cout << "Set contains:";
    while (!s.empty()) {
        std::cout << ' ' << *s.begin();
        s.erase(s.begin());
    }
    std::cout << std::endl;

    return 0;
}
