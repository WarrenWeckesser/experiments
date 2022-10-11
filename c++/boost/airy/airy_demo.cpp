#include <iostream>
#include <iomanip>
#include <boost/math/special_functions/airy.hpp>

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
    std::cout << std::setw(w) << "ai";
    std::cout << std::setw(w) << "ai'";
    std::cout << std::setw(w) << "bi";
    std::cout << std::setw(w) << "bi'";
    std::cout << std::endl;
    for (const auto& x: v) {
        double ai = airy_ai(x);
        double aip = airy_ai_prime(x);
        double bi = airy_bi(x);
        double bip = airy_bi_prime(x);
        std::cout << std::setw(18) << std::setprecision(12) << x;
        std::cout << std::setw(w) << std::setprecision(17) << ai;
        std::cout << std::setw(w) << std::setprecision(17) << aip;
        std::cout << std::setw(w) << std::setprecision(17) << bi;
        std::cout << std::setw(w) << std::setprecision(17) << bip;
        std::cout << std::endl;
    }

    return 0;
}
