
#include <iostream>
#include <random>
#include <cmath>


double next_greater(double x, size_t n = 1)
{
    double y = x;
    for (int k = 0; k < n; ++k) {
        y = std::nextafter(y, INFINITY);
    }
    return y;
}


int main()
{
    double a = 0.0;
    double b = next_greater(a, 10);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);
    int n = 100;
    int count = 0;
    for (int k = 0; k < n; ++k) {
        double r = dis(gen);
        if (r == b) {
            ++count;
        } 
    }
    std::cout << "b occurred " << count << " times  in " << n << " samples.\n";
}
