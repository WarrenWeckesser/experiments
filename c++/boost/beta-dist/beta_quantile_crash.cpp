#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;

typedef boost::math::policies::policy<
    boost::math::policies::promote_float<false>,
    boost::math::policies::promote_double<false>
> MyPolicy;

template<typename T>
T beta_quantile(T p, T a, T b)
{
    beta_distribution<T, MyPolicy> dist(a, b);
    T x = quantile(dist, p);
    return x;    
}

template<typename T>
void checkit()
{
    T a = 5.0;
    T b = a;
    T p = 0.5;
 
    T x = beta_quantile(p, a, b);
    cout << scientific << setw(23) << setprecision(16) << x << endl;
}

int main(int argc, char *argv[])
{
    checkit<float>();
    checkit<double>();

    return 0;
}
