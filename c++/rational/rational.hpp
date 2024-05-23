
namespace rational {

//
// fixed_rational(x) evaluates a fixed rational function. It is
// assumed that the coefficients were computed in advance, so they are
// incorporated as constant arrays within the function.
//

double fixed_rational_loop(const double x)
{
    const double P[] = {1.0, 0.5, 0.125, -0.125, 0.0, 0.5, 0.0, -0.05};
    const double Q[] = {1.0, 0.25, 0.375, 0.125, 0.5, 0.5, -0.5, 0.25};
    double num = P[7];
    double den = Q[7];
    for (std::size_t i = 7; i > 0; --i) {
        num = num*x + P[i-1];
        den = den*x + Q[i-1];
    }
    return num/den;
}

double fixed_rational(const double x)
{
    // const double P[] = {1.0, 0.5, 0.125, -0.125, 0.0, 0.5, 0.0, -0.05};
    // const double Q[] = {1.0, 0.25, 0.375, 0.125, 0.5, 0.5, -0.5, 0.25};
    double num = 1.0 + x*(0.5 + x*(0.125 + x*(-0.125 + x*(0.0 + x*(0.5 + x*(0.0 + x*(-0.05)))))));
    double den = 1.0 + x*(0.25 + x*(0.375 + x*(0.125 + x*(0.5 + x*(0.5 + x*(-0.5 + x*(0.25)))))));
    return num/den;
}

//
// Generic functions for polynomial and rational functions.
//
// In the vectors of polynomial coefficients, the first coefficient
// is the constant term, the second is for the linear term, etc.
// For example, if P = [1.0, 2.5, 0.5], then the polynomial function
// is 1.0 + 2.5*x + 0.5*x**2.
//

template<typename T>
T eval_poly(const T x, const std::vector<T>& P)
{
    std::size_t n = P.size();
    T y = P[n - 1];
    for (std::size_t i = n - 1; i > 0; --i) {
        y = y*x + P[i - 1];
    }
    return y;
}

template<typename T>
T eval_poly_reversed(const T x, const std::vector<T>& P)
{
    std::size_t n = P.size();
    T y = P[0];
    for (std::size_t i = 1; i < n; ++i) {
        y = y*x + P[i];
    }
    return y;
}


template<typename T>
T eval_rational(const T x, const std::vector<T>& P, const std::vector<T>& Q)
{
    return eval_poly(x, P) / eval_poly(x, Q);
}

template<typename T>
T eval_rational_with_inversion(const T x, const std::vector<T>& P, const std::vector<T>& Q)
{
    if (x <= 1) {
        return eval_poly(x, P) / eval_poly(x, Q);
    }
    else {
        T z = 1/x;
        return eval_poly_reversed(z, P) / eval_poly_reversed(z, Q);
    }
}


}  // namespace rational
