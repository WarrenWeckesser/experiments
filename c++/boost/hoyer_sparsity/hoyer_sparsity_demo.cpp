#include <iostream>
#include <string>
#include <boost/math/statistics/signal_statistics.hpp>


int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "use: " << argv[0] << " x0 x1 x2 ...\n";
        std::cout << "At least two values must be provided." << std::endl;
        return -1;
    }

    std::vector<double> x;

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
        x.push_back(arg);
    }

    double h = boost::math::statistics::hoyer_sparsity(x);
    std::cout << std::setprecision(9) << "h = " << h << std::endl;

    return 0;
}
