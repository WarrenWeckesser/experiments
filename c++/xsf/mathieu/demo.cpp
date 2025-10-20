#include <iomanip>
#include <iostream>
#include <numbers>
#include <xsf/mathieu.h>

int main(int argc, char *argv[])
{
    using std::numbers::pi_v;
    double f, df;

    double m = 3.0;
    double q = 1.5;
    double theta = 1.0;                     // radians
    double angle = theta*180/pi_v<double>;  // degrees
    xsf::cem(m, q, angle, f, df);

    std::cout << std::setprecision(15) << "f  = " << f << std::endl;
    std::cout << std::setprecision(15) << "df = " << df << std::endl;

    return 0;
}
