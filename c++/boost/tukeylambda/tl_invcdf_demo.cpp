#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <chrono>

#include "tukeylambda.h"

using boost::multiprecision::cpp_bin_float_100;

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using Clock = std::chrono::steady_clock;
	using Second = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<Clock> m_beg { Clock::now() };

public:
	void reset()
	{
		m_beg = Clock::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
	}
};

//
// Internally, this function uses the type `cpp_bin_float_100` from boost/multiprecision
// to compute the result.  `cpp_bin_float_100` provides 50 digits of precision.
//
// I haven't tested the limits, but this can still result in low precision outputs of
// this function in extreme cases.  `cpp_bin_float_100` is still finite precision,
// and subject to subtractive loss of precision.
//
double invcdf_mp(double p, double lam)
{
    cpp_bin_float_100 p_mp = p;
    cpp_bin_float_100 lam_mp = lam;
    cpp_bin_float_100 x_mp;
    if (lam == 0) {
        x_mp = log(p_mp) - log1p(-p_mp);
    }
    else {
        x_mp = (pow(p_mp, lam_mp) - pow(1 - p_mp, lam_mp))/lam_mp;
    }
    double x = static_cast<double>(x_mp);
    return x;
}

void timing(int n, double lam)
{
    std::vector<double> p(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Fill p with uniform random values from the interval [0, 1).
    for (int k = 0; k < n; ++k) {
        p[k] = dis(gen);
    }

    Timer t;
    double sum;
    for (int k = 0; k < n; ++k) {
        sum += tukey_lambda_invcdf(p[k], lam);
    }
    double duration = t.elapsed();
    printf("sum = %25.15e\n", sum);
    printf("duration = %15.9f second\n", duration);

    sum = 0;
    t.reset();
    for (int k = 0; k < n; ++k) {
        sum += tukey_lambda_invcdf_experimental(p[k], lam);
    }
    duration = t.elapsed();
    printf("sum = %25.15e\n", sum);
    printf("duration = %15.9f second\n", duration);
}

int main()
{
    double p = 1e-8;
    double lam = -3e-50;
    printf("p = %25.16e   lam = %25.16e\n", p, lam);

    double invcdf1 = tukey_lambda_invcdf(p, lam);
    printf("invcdf (standard)       = %25.16e\n", invcdf1);

    double invcdf2 = tukey_lambda_invcdf_experimental(p, lam);
    printf("invcdf (experimental)   = %25.16e\n", invcdf2);

    double invcdfmp = invcdf_mp(p, lam);
    printf("invcdf (multiprecision) = %25.16e\n", invcdfmp);

    printf("\n");
    timing(1000000, lam);
}
