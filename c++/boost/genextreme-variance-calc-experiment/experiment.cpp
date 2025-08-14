
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

using namespace boost::math::policies;
using boost::math::tgamma;
using namespace std;


//typedef policy<underflow_error<throw_on_error>> my_policy;
typedef policy<promote_double<false>> no_double_promotion;

double var1(double c)
{
    double g1 = tgamma(1 - c, no_double_promotion());
    double g2 = tgamma(1 - 2*c, no_double_promotion());
    return (g2 - g1*g1)/(c*c); 
}

double var2(double c)
{
    double g1 = tgamma(1 - c, no_double_promotion());
    double g2 = tgamma(1 - 2*c, no_double_promotion());
    double sqrtg2 = sqrt(g2);
    return (sqrtg2 - g1)*(sqrtg2 + g1)/c/c;
}

int main()
{
    double c;
 
    for(c = 0.125; c > -0.125; c-= 0.001) {
        double g1 = tgamma(1 - c, no_double_promotion());
        double g2 = tgamma(1 - 2*c, no_double_promotion());
        double y1 = var1(c);
        double y2 = var2(c);
        cout << setprecision(5) << setw(9) << c
             << setprecision(17) << setw(26) << y1
             << setprecision(17) << setw(26) << y2
             << endl;
    }
}
