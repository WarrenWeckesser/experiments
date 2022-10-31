#include <iostream>
#include <iomanip>
#include "boost/math/special_functions/lambert_w.hpp"

using namespace boost::math;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "use: " << argv[0] << " x0 x1 x2 ...\n";
        std::cout << "At least one value must be provided." << std::endl;
        return -1;
    }

    std::vector<double> v;

    for (int k = 1; k < argc; ++k) {
        double arg;
        try {
            arg = std::stod(argv[k]);
        }
        catch(...) {
            std::cout << "Error: unable to convert '" << argv[k]
                      << "' to double." << std::endl;
            return -1;
        }
        v.push_back(arg);
    }

    int wx = 18;
    int w = 25;
    std::cout << std::setw(wx) << "x";
    std::cout << std::setw(w) << "W{0}(x)";
    std::cout << std::setw(w) << "W{-1}(x)";
    std::cout << std::endl;
    double bound = -std::exp(-1);
    for (const auto& x: v) {
        double w0 = x >= bound ? lambert_w0(x) : NAN;
        double wm1 = (x >= bound && x < 0) ? lambert_wm1(x) : NAN;
        std::cout << std::setw(18) << std::setprecision(12) << x;
        std::cout << std::setw(w) << std::setprecision(17) << w0;
        std::cout << std::setw(w) << std::setprecision(17) << wm1;
        std::cout << std::endl;
    }

    return 0;
}
