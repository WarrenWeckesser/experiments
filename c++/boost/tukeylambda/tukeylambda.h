#include <cstdio>
#include <cmath>
#include <limits>
#include <array>

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/logaddexp.hpp>

//
// In toms748_solve, don't promote double to long double, so we
// can have consistent behavior across different platforms.
//
typedef boost::math::policies::policy<
    boost::math::policies::promote_double<false>
> Toms748Policy;

//
// Inverse of the CDF of the logistic distribution:
//   x = log(p/(1 - p))
// This function assumes 0 <= p <= 1.
//
double logistic_invcdf(double p)
{
    if (p == 0) {
        return -INFINITY;
    }
    if (p == 1) {
        return INFINITY;
    }
    if (p < 0.3 || p > 0.61) {
        return std::log(p/(1 - p));
    }
    // Let d = p - 0.5, so p = d + 0.5.
    // Then
    //      log(p/(1 - p)) = log((0.5 + d)/(0.5 - d))
    //                     = log((1 + 2*d)/(1 - 2*d))
    //                     = log(1 + 2*d) - log(1 - 2*d)
    //                     = log1p(2*d) - log1p(-2*d)
    double s = 2*(p - 0.5);
    return std::log1p(s) - std::log1p(-s);
}


//
// Handle the edges cases of the Tukey lambda distribution inverse cdf.
// If the calculation is handled by this function, *result is assigned
// and 1 is returned.  Otherwise 0 is returned.
//
int tukey_lambda_invcdf_edge_case(double p, double lam, double *result)
{
    if (std::isnan(p) || !std::isfinite(lam) || p < 0 || p > 1) {
        *result = std::numeric_limits<double>::quiet_NaN();
        return 1;
    }
    if (p == 0) {
        *result = lam <= 0 ? -std::numeric_limits<double>::infinity() : -1/lam;
        return 1;
    }
    if (p == 1) {
        *result = lam <= 0 ? std::numeric_limits<double>::infinity() : 1/lam;
        return 1;
    }
    if (p == 0.5) {
        return 0.0;
    }
    if (lam == 0) {
        *result = logistic_invcdf(p);
        return 1;
    }
    const double log_max = std::log(std::numeric_limits<double>::max());
    if (lam < 0) {
        if (p > 0.5 && lam*std::log1p(-p) > log_max) {
            // Overflow of (1 - p)**lam.
            *result = std::numeric_limits<double>::infinity();
            return 1;
        }
        if (p < 0.5 && lam*std::log(p) > log_max) {
            // Overflow of p**lam.
            *result = -std::numeric_limits<double>::infinity();
            return 1;
        }
    }
    return 0;
}

//
// Inverse of the CDF of the Tukey lambda distribution.
// This is the direct implementation of the formula, i.e. the "naive"
// implementation. ("Naive" doesn't mean "bad"; it works fine for a wide
// range of parameter values.)
//
double tukey_lambda_invcdf(double p, double lam)
{
    double x;
    if (tukey_lambda_invcdf_edge_case(p, lam, &x)) {
        return x;
    }
    return (std::pow(p, lam) - std::pow(1 - p, lam))/lam;
}

static inline double
pow1p(double x, double r)
{
    return std::exp(r*std::log1p(x));
}

double tukey_lambda_invcdf2(double p, double lam)
{
    double x;
    if (tukey_lambda_invcdf_edge_case(p, lam, &x)) {
        return x;
    }
    if (p > 0.3 && p < 0.7) {
        double t = 2*p - 1.0;
        x = std::pow(2.0, -lam)*(pow1p(t, lam) - pow1p(-t, lam))/lam;
    }
    else {
        x = (std::pow(p, lam) - std::pow(1 - p, lam))/lam;
    }
    return x;
}

double tukey_lambda_invcdf3a(double p, double lam)
{
    double x;
    if (tukey_lambda_invcdf_edge_case(p, lam, &x)) {
        return x;
    }
    double t = lam*logistic_invcdf(p);
    return -std::pow(p, lam) * std::expm1(-t)/lam;
}


double tukey_lambda_invcdf3b(double p, double lam)
{
    double x;
    if (tukey_lambda_invcdf_edge_case(p, lam, &x)) {
        return x;
    }
    double t = lam*logistic_invcdf(p);
    return pow1p(-p, lam)*std::expm1(t)/lam;
}

#define TUKEY_LAMBDA_INVCDF_TAYLOR_MAX_N 31

double tukey_lambda_invcdf_taylor(double p, double lam, int n)
{
    double x;
    if (tukey_lambda_invcdf_edge_case(p, lam, &x)) {
        return x;
    }

    if (n > TUKEY_LAMBDA_INVCDF_TAYLOR_MAX_N) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    x = logistic_invcdf(p);
    // lam == 0 is handled above by tukey_lambda_edge_case(), so
    // so we know lam != 0 here.
    std::array<double, TUKEY_LAMBDA_INVCDF_TAYLOR_MAX_N+1> logp_powers, logq_powers;
    double logp = std::log(p);
    double logq = std::log1p(-p);
    double logp_power = 1.0;
    double logq_power = 1.0;
    for (int k = 0; k <= n; ++k) {
        logp_powers[k] = logp_power;
        logq_powers[k] = logq_power;
        logp_power *= logp;
        logq_power *= logq;
    }
    double sum = 0.0;
    double fac = 1.0;
    double lamp = 1.0;
    for (int k = 0; k <= n; ++k) {
        double c = lamp/fac;
        double inner_sum = 0.0;
        for (int j = 0; j <= k; ++j) {
            inner_sum += logp_powers[k - j]*logq_powers[j];
        }
        sum += c*inner_sum;
        lamp *= lam;
        fac *= k + 2.0;
    }
    x *= sum;
    return x;
}

//
// Inverse of the CDF of the Tukey lambda distribution--
// experimental version.
//
double tukey_lambda_invcdf_experimental(double p, double lam)
{
    double x;
    if (tukey_lambda_invcdf_edge_case(p, lam, &x)) {
        return x;
    }

    x = logistic_invcdf(p);
    double q = 1 - p;
    int k = 1;
    while (k < 100) {
        lam = lam/2.0;
        double f = 0.5*(std::pow(p, lam) + std::pow(q, lam));
        if (f == 1.0) {
            // printf("=== k = %d  p = %23.16e  x = %23.16e\n", k, p, x);
            return x;
        }
        double nextx = x*f;
        x = nextx;
        ++k;
    }
    // This sometimes happens with extreme values of the inputs.
    // This needs more investigation.
    printf("*** k = %d  p = %23.16e  x = %23.16e\n", k, p, x);
    return std::numeric_limits<double>::quiet_NaN() ;
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
            if (x < -std::pow(2.0, 1 - lam)) {
                pmin = 0.0;
            }
            else {
                pmin = 0.5 + std::pow(2.0, lam - 2)*x;
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
            if (x > std::pow(2.0, 1 - lam)) {
                pmax = 1.0;
            }
            else {
                pmax = 0.5 + std::pow(2.0, lam - 2)*x;
            }
        }
    }
    else {
        pmin = 0.0;
        pmax = 1.0;
    }
    return std::pair(pmin, pmax);
}

inline double
inf_to_big(double x)
{
    return std::isinf(x) ? 0.75*std::copysign(std::numeric_limits<double>::max(), x)
                         : x;
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
    printf("  :: initial_bracket: %25.16e %25.16e\n", initial_bracket.first, initial_bracket.second);
    double x_second = tukey_lambda_invcdf(initial_bracket.second, lam);

    if (initial_bracket.first == 0.0) {
        if (x_second <= x) {
            // Extreme case: because of floating point imprecision, the
            // interval doesn't actually bracket the root.  This is observed
            // to occur in the far left tail, so we can use the upper end of
            // the bracket as the result.
            return initial_bracket.second;
        }
    }
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
                                // toms748_solve doesn't handle inf.
                                return inf_to_big(x1);
                            },
                            initial_bracket.first, initial_bracket.second,
                            tol, max_iter, Toms748Policy());
    printf("  :: bracket:         %25.16e %25.16e\n", bracket.first, bracket.second);

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
    // We need the CDF and the SF.  Compute the smaller one, and subtract it
    // from 1 to get the larger.
    double p_smaller;
    if (x < 0) {
        p_smaller = tukey_lambda_cdf(x, lam);
    }
    else {
        p_smaller = tukey_lambda_sf(x, lam);
    }
    //double tmp =  std::pow(p_smaller, lam - 1) + std::pow(1 - p_smaller, lam - 1);
    double denom = std::pow(p_smaller, lam - 1) + std::exp((lam - 1)*std::log1p(-p_smaller));
    return 1/denom;
}

//
// Natural logarithm of the PDF of the Tukey lambda distribution.
//
double tukey_lambda_logpdf(double x, double lam)
{
    if (std::isnan(x) || !std::isfinite(lam)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (lam > 0) {
        double recip_lam = 1.0/lam;
        if (x < -recip_lam || x > recip_lam) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    double p_smaller;
    if (x < 0) {
        p_smaller = tukey_lambda_cdf(x, lam);
    }
    else {
        p_smaller = tukey_lambda_sf(x, lam);
    }
    // The return expression is equivalent to
    //   -log(bigger**(lam - 1) + smaller**(lam - 1))
    return -boost::math::logaddexp((lam - 1)*std::log1p(-p_smaller),
                                   (lam - 1)*std::log(p_smaller));
}
