
#include <cmath>
#include <limits>

#include <boost/math/tools/roots.hpp>


//
// In toms748_solve, don't promote double to long double,
// so we can have consistent behavior that does not depend
// on platform-dependent implementations of long double.
//
typedef boost::math::policies::policy<
    boost::math::policies::promote_double<false>
> Toms748Policy;


//
// Inverse of the CDF of the Tukey lambda distribution.
//
double tukey_lambda_invcdf(double p, double lam)
{
    double x;

    if (std::isnan(p) || !std::isfinite(lam) || p < 0 || p > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (p == 0) {
        x = lam <= 0 ? -INFINITY : -1/lam;
    }
    else if (p == 1) {
        x = lam <= 0 ? INFINITY : 1/lam;
    }
    else {
        if (lam == 0) {
            x = std::log(p) - std::log1p(-p);
        }
        else {
            x = (std::pow(p, lam) - std::pow(1 - p, lam))/lam;
        }
    }
    return x;
}

//
// Inverse of the survival function of the Tukey lambda distribution.
//
double tukey_lambda_invsf(double p, double lam)
{
    return -tukey_lambda_invcdf(p, lam);
}


std::pair<double, double>
get_cdf_solver_bracket(double x, double lam)
{
    double pmin, pmax;

    if (lam < 0) {
        if (x < 0) {
            if (x < std::pow(2.0, -lam)/lam) {
                pmax = std::pow(lam*x, 1/lam);
            }
            else {
                pmax = 0.5;
            }
            if (x < -std::pow(2.0, -lam - 1.0)) {
                pmin = 0.0;
            }
            else {
                pmin = 0.5 + std::pow(2.0, lam)*x;
            }
        }
        else {
            // x > 0
            if (x > -std::pow(2.0, -lam)/lam) {
                pmin = -std::expm1(std::log(-lam*x)/lam);
            }
            else {
                pmin = 0.5;
            }
            if (x > std::pow(2.0, -lam - 1.0)) {
                pmax = 1.0;
            }
            else {
                pmax = 0.5 + std::pow(2.0, lam)*x;
            }
        }
    }
    else {
        pmin = 0.0;
        pmax = 1.0;
    }
    return std::pair(pmin, pmax);
}

//
// Cumulative distribution function of the Tukey lambda distribution.
//
double tukey_lambda_cdf(double x, double lam)
{
    if (std::isnan(x) || !std::isfinite(lam)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(x)) {
        if (x > 0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    if (lam > 0) {
        // When lam > 0, support is [-1/lam, 1/lam].
        if (x >= 1/lam) {
            return 1.0;
        }
        if (x <= -1/lam) {
            return 0.0;
        }
    }
    if (lam == 0) {
        if (x >= 0) {
            return 1.0/(1.0 + std::exp(-x));
        } else {
            return std::exp(x)/(1.0 + std::exp(x));
        }
    }

    std::pair<double, double> initial_bracket = get_cdf_solver_bracket(x, lam);
    if (initial_bracket.first == 1.0) {
        return 1.0;
    }
    if (initial_bracket.second == 0.0) {
        return 0.0;
    }

    boost::math::tools::eps_tolerance<double> tol;
    std::pair<double, double> bracket;
    std::uintmax_t max_iter = 100;
    bracket = toms748_solve([x, lam](double p) {
                                double x1 = tukey_lambda_invcdf(p, lam) - x;
                                // Apparently toms748_solve does not handle inf, so
                                // translate infs to very big numbers.
                                if (std::isinf(x1)) {
                                    x1 = std::copysign(std::numeric_limits<double>::max(), x1);
                                }
                                return x1;
                            },
                            initial_bracket.first, initial_bracket.second,
                            tol, max_iter, Toms748Policy());
    if (bracket.first == bracket.second) {
        return bracket.first;
    }
    else {
        // In some experiments, the result was better at an endpoint of the
        // interval than in the middle, so here the endpoint with the better
        // result is chosen to be the return value.
        double x1 = tukey_lambda_invcdf(bracket.first, lam);
        double x2 = tukey_lambda_invcdf(bracket.second, lam);
        if (std::abs(x - x1) < std::abs(x - x2)) {
            return bracket.first;
        }
        else {
            return bracket.second;
        }
    }
}


//
// Survival function of the Tukey lambda distribution.
//
double tukey_lambda_sf(double x, double lam)
{
    return tukey_lambda_cdf(-x, lam);
}


//
// Probability density function of the Tukey lambda distribution.
//
double tukey_lambda_pdf(double x, double lam)
{
    if (std::isnan(x) || !std::isfinite(lam)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (lam > 0) {
        double recip_lam = 1.0/lam;
        if (x < -recip_lam || x > recip_lam) {
            return 0.0;
        }
    }
    double cdf = tukey_lambda_cdf(x, lam);
    double sf = tukey_lambda_sf(x, lam);
    double p = 1/(std::pow(cdf, lam - 1) + std::pow(sf, lam - 1));
    return p;
}
