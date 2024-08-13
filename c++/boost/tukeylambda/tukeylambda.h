
#include <cmath>
#include <cstdio>
#include <limits>

#include <boost/math/tools/roots.hpp>


//
// In toms748_solver, don't promote double to long double,
// so we can have consistent behavior that does not depend
// on the platform-dependent implementation of long double.
//
typedef boost::math::policies::policy<
    //boost::math::policies::promote_float<false>,
    boost::math::policies::promote_double<false>
> Toms748Policy;


//
//  This function assumes:
//  * p and lam are finite
//  * 0 <= p <= 1
//  * lam != 0
//
double tukey_lambda_invcdf(double p, double lam)
{
    double x;
    // The values +/-1.7e308 are replacements for infinity.
    // These are a hack to make toms748_solver work; it did
    // not like getting infinity.
    if (p == 0) {
        x = lam < 0 ? - 1.7e308: -1/lam;
    }
    else if (p == 1) {
        x = lam < 0 ? 1.7e308 : 1/lam;
    }
    else {
        // This expression might need to be reformulated.
        x = (std::pow(p, lam) - std::pow(1 - p, lam))/lam;
    }
    //printf("  x = %25.15e\n", x);
    return x;
}

//
//  This function assumes:
//  * p and lam are finite
//  * 0 <= p <= 1
//  * lam != 0
//
double tukey_lambda_invsf(double p, double lam)
{
    double x;
    // The values +/-1.7e308 are replacements for infinity.
    // These are a hack to make toms748_solver work; it did
    // not like getting infinity.
    if (p == 0) {
        x = lam < 0 ? 1.7e308: 1/lam;
    }
    else if (p == 1) {
        x = lam < 0 ? -1.7e308 : -1/lam;
    }
    else {
        // This expression might need to be reformulated.
        x = (std::pow(1 - p, lam) - std::pow(p, lam))/lam;
    }
    //printf("  x = %25.15e\n", x);
    return x;
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
    printf("initial_bracket = (%26.16e, %26.16e)\n",
           initial_bracket.first, initial_bracket.second);
    if (initial_bracket.first == 1.0) {
        return 1.0;
    }
    if (initial_bracket.second == 0.0) {
        return 0.0;
    }

    boost::math::tools::eps_tolerance<double> tol;
    std::pair<double, double> bracket;
    std::uintmax_t max_iter = 100;
    bracket = toms748_solve([x, lam](double p) {return tukey_lambda_invcdf(p, lam) - x;},
                            initial_bracket.first, initial_bracket.second,
                            tol, max_iter, Toms748Policy());
    printf("bracket = (%26.16e, %26.16e)\n", bracket.first, bracket.second);
    if (bracket.first == bracket.second) {
        return bracket.first;
    }
    else {
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

double tukey_lambda_sf(double x, double lam)
{
    if (std::isnan(x) || !std::isfinite(lam)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(x)) {
        if (x > 0) {
            return 0.0;
        }
        else {
            return 1.0;
        }
    }
    if (lam > 0) {
        // When lam > 0, support is [-1/lam, 1/lam].
        if (x >= 1/lam) {
            return 0.0;
        }
        if (x <= -1/lam) {
            return 1.0;
        }
    }
    if (lam == 0) {
        if (x >= 0) {
            return std::exp(-x)/(1.0 + std::exp(-x));
        } else {
            return 1.0/(1.0 + std::exp(x));
        }
    }

    std::pair<double, double> initial_bracket = get_cdf_solver_bracket(-x, lam);
    printf("initial_bracket = (%26.16e, %26.16e)\n",
           initial_bracket.first, initial_bracket.second);
    if (initial_bracket.first == 1.0) {
        return 1.0;
    }
    if (initial_bracket.second == 0.0) {
        return 0.0;
    }

    boost::math::tools::eps_tolerance<double> tol;
    std::pair<double, double> bracket;
    std::uintmax_t max_iter = 80;
    bracket = toms748_solve([x, lam](double p) {return tukey_lambda_invsf(p, lam) - x;},
                            initial_bracket.first, initial_bracket.second,
                            tol, max_iter, Toms748Policy());
    printf("bracket = (%26.16e, %26.16e)\n", bracket.first, bracket.second);
    if (bracket.first == bracket.second) {
        return bracket.first;
    }
    else {
        double x1 = tukey_lambda_invsf(bracket.first, lam);
        double x2 = tukey_lambda_invsf(bracket.second, lam);
        if (std::abs(x - x1) < std::abs(x - x2)) {
            return bracket.first;
        }
        else {
            return bracket.second;
        }
    }
}
