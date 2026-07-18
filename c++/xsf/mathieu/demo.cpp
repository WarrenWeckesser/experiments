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

    std::cout << std::setprecision(15);
    std::cout << "f = " << f << std::endl;
    std::cout << "df          = " << df << std::endl;

    double h = 1e-8;
    double fplus, fminus, tmp;
    xsf::cem(m, q, angle + h, fplus, tmp);
    xsf::cem(m, q, angle - h, fminus, tmp);
    double approx_diff = (fplus - fminus)/(2*h);
    std::cout << "finite diff = " << approx_diff << std::endl;
    std::cout << "df*π/180    = " << df * pi_v<double> / 180.0 << std::endl;

    return 0;
}
