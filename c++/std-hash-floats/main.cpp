#include <complex>
// For std::hash:
#include <functional>
#include <iostream>
#include <limits>

int main()
{
    double x = std::numeric_limits<double>::quiet_NaN();
    auto hash_x = std::hash<double>{}(x);
    std::cout << "hash(x)  = " << hash_x << std::endl;

    double y = std::numeric_limits<double>::quiet_NaN();
    auto hash_y = std::hash<double>{}(y);
    std::cout << "hash(y)  = " << hash_y << std::endl;

    long double lx = std::numeric_limits<long double>::quiet_NaN();
    auto hash_lx = std::hash<double>{}(lx);
    std::cout << "hash(lx) = " << hash_lx << std::endl;

    double pos_zero = 0.0;
    auto hash_pos_zero = std::hash<double>{}(pos_zero);
    std::cout << "hash(+0.0) = " << hash_pos_zero << std::endl;

    double neg_zero = -0.0;
    auto hash_neg_zero = std::hash<double>{}(neg_zero);
    std::cout << "hash(-0.0) = " << hash_neg_zero << std::endl;

    return 0;
}
