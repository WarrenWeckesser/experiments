#include <iostream>
#include <string>
#include <boost/math/statistics/signal_statistics.hpp>


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("use: %s x0 x1 x2 ...\n", argv[0]);
        exit(-1);
    }
    std::vector<double> x;

    for (int k = 1; k < argc; ++k) {
        double arg = std::stod(argv[k]);
        x.push_back(arg);
    }

    double h = boost::math::statistics::hoyer_sparsity(x);
    std::cout << std::setprecision(9) << "h = " << h << std::endl;

    return 0;
}
