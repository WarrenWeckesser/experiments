#include <iostream>
#include <boost/math/distributions/students_t.hpp>

using namespace std;
using boost::math::students_t;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "use: " << argv[0] << " dof p\n";
        return -1;
    }
    cout << "Student's t distribution, quantile function. " << endl;

    double dof = std::stod(argv[1]);
    double p = std::stod(argv[2]);

    cout << "dof = " << dof << endl;
    cout << "p = " << p << endl;

    auto dist = students_t(dof);
    double x = quantile(dist, p);

    cout << "quantile: x = " << setw(18) << setprecision(15) << x << endl;

    return 0;
}