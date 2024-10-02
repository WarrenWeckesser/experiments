#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <chrono>

#include "tukeylambda.h"

using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_bin_float_50;

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
// Uses type T (e.g. cpp_bin_float_50, cpp_bin_float_100) internally
// for floating point calculations.
//
template<typename T>
double invcdf_mp(double p, double lam)
{
    T p_mp = p;
    T lam_mp = lam;
    T x_mp;
    if (lam == 0) {
        x_mp = log(p_mp) - log1p(-p_mp);
    }
    else {
        x_mp = (pow(p_mp, lam_mp) - pow(1 - p_mp, lam_mp))/lam_mp;
    }
    return static_cast<double>(x_mp);
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
    // double p = 0.501;
    // double lam = -3e-40;

    double p = 0.48;
    double lam = -0.1;
    printf("p = %25.16e   lam = %25.16e\n", p, lam);

    double invcdf1 = tukey_lambda_invcdf(p, lam);
    printf("invcdf (standard)          = %25.16e\n", invcdf1);

    double invcdf2 = tukey_lambda_invcdf2(p, lam);
    printf("invcdf (version 2)         = %25.16e\n", invcdf2);

    // double eps = p - 0.5;
    // double invcdfla = std::pow(2.0, 2 - lam)*eps;
    // printf("invcdf (linear p - 0.5)    = %25.16e\n", invcdfla);

    int order = 3;
    double invcdft3 = tukey_lambda_invcdf_taylor(p, lam, order);
    printf("invcdf (taylor %3d)        = %25.16e\n", order, invcdft3);

    order = 9;
    double invcdft9 = tukey_lambda_invcdf_taylor(p, lam, order);
    printf("invcdf (taylor %3d)        = %25.16e\n", order, invcdft9);

    order = 12;
    double invcdft12 = tukey_lambda_invcdf_taylor(p, lam, order);
    printf("invcdf (taylor %3d)        = %25.16e\n", order, invcdft12);

    order = 15;
    double invcdft15 = tukey_lambda_invcdf_taylor(p, lam, order);
    printf("invcdf (taylor %3d)        = %25.16e\n", order, invcdft15);

    double invcdfx = tukey_lambda_invcdf_experimental(p, lam);
    printf("invcdf (experimental)      = %25.16e\n", invcdfx);

    double invcdfmp50 = invcdf_mp<cpp_bin_float_50>(p, lam);
    printf("invcdf (cpp_bin_float_50)  = %25.16e\n", invcdfmp50);

    double invcdfmp100 = invcdf_mp<cpp_bin_float_100>(p, lam);
    printf("invcdf (cpp_bin_float_100) = %25.16e\n", invcdfmp100);

    printf("\n");
    timing(1000000, lam);
}
