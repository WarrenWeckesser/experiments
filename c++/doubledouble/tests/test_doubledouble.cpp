
#include <cstdio>
#include "checkit.h"
#include "doubledouble.h"

//
// Some of the equality tests of the lower parts of the results might be
// too optimistic.  They might fail on different combinations of platforms,
// compilers, and compiler options.
//

int main(int argc, char *argv[])
{
    auto test = CheckIt(std::cerr);

    auto three = DoubleDouble(3.0);
    auto x = DoubleDouble(10.0, 3e-18);
    auto zero = DoubleDouble(0.0, 0.0);
    auto one = DoubleDouble(1.0, 0.0);
    auto p6 = DoubleDouble(3.14159, 0.0);
    auto t = DoubleDouble(3.7500950075008e-15, 0.0);

    check_equal_fp(test, three.upper, 3.0, "Simple test of constructor: upper");
    check_equal_fp(test, three.lower, 0.0, "Simple test of constructor: lower");

    check_equal_fp(test, x.upper, 10.0, "Simple test of constructor: upper");
    check_equal_fp(test, x.lower, 3e-18, "Simple test of constructor: lower");

    auto y = 1.0 / three;
    check_equal_fp(test, y.upper, 0.3333333333333333, "1/3 (upper)");
    check_equal_fp(test, y.lower, 1.850371707708594e-17, "1/3 (lower)");

    y = x*x;
    check_equal_fp(test, y.upper, 100.0, "Check x*x (upper)");
    check_equal_fp(test, y.lower, 6e-17, "Check x*x (lower)");

    y = one - DoubleDouble(2.0)/3;
    check_equal_fp(test, y.upper, 0.3333333333333333, "Check 1 - 2/3 (upper)");
    check_close_fp(test, y.lower, 1.850371707708594e-17, 5e-16, "Check 1 - 2/3 (lower)");

    y = x.log();
    check_equal_fp(test, y.upper, 2.302585092994046, "Check log(x) (upper)");
    check_equal_fp(test, y.lower, -2.1677562233822494e-16, "Check log(x) (lower)");

    y = x.powi(4);
    check_equal_fp(test, y.upper, 10000.0, "Check x**4 (upper)");
    check_close_fp(test, y.lower, 1.2e-14, 5e-16, "Check x**4 (lower)");

    y = (dd_pi - p6)*((dd_pi + p6)/2);
    check_equal_fp(test, y.upper, 8.336494679678182e-06, "Check (pi - p6)*((pi + p6)/2) (upper)");
    check_close_fp(test, y.lower, -1.4573851200099609e-22, 5e-16, "Check (pi - p6)*((pi + p6)/2) (lower)");

    y = zero.exp();
    check_equal_fp(test, y.upper, 1.0, "exp(0) (upper)");
    check_equal_fp(test, y.lower, 0.0, "exp(0) (lower)");

    y = one.exp();
    check_equal_fp(test, y.upper, 2.718281828459045, "exp(0) (upper)");
    check_equal_fp(test, y.lower, 1.4456468917292502e-16, "exp(0) (lower)");

    y = x.exp();
    check_equal_fp(test, y.upper, 22026.465794806718, "exp(x) (upper)");
    check_equal_fp(test, y.lower, -1.3119340726673171e-12, "exp(x) (lower)");

    auto d = x - three;
    check_equal_fp(test, d.upper, 7.0, "x - 3 (upper)");
    check_equal_fp(test, d.lower, 3e-18, "x - 3 (lower)");

    d = three - x;
    check_equal_fp(test, d.upper, -7.0, "3 - x (upper)");
    check_equal_fp(test, d.lower, -3e-18, "3 - x (lower)");

    d = d.abs();
    check_equal_fp(test, d.upper, 7.0, "abs(3 - x) (upper)");
    check_equal_fp(test, d.lower, 3e-18, "abs(3 - x) (lower)");

    y = x - 10;
    check_equal_fp(test, y.upper, 3e-18, "x - 10 (upper)");
    check_equal_fp(test, y.lower, 0.0, "x - 10 (lower)");

    y = dd_pi.exp();
    check_equal_fp(test, y.upper, 23.14069263277927, "exp(pi) (upper)");
    check_equal_fp(test, y.lower, -1.3488747091995788e-15, "exp(pi) (lower)");

    y = dd_e.log();
    check_equal_fp(test, y.upper, 1.0, "log(e) (upper)");
    check_equal_fp(test, y.lower, 0.0, "log(e) (lower)");

    // The actual relative error for log(1.000000099005) is approx. 8.7e-10.
    // Can we do better?
    y = DoubleDouble(1.000000099005, 0.0).log();
    check_equal_fp(test, y.upper, 9.900499507506536e-08, "log(1.000000099005) (upper)");
    check_close_fp(test, y.lower, 4.563816054961034e-24, 2e-9, "log(1.000000099005) (lower)");

    y = (x - 10.0).exp();
    check_equal_fp(test, y.upper, 1.0, "exp(x - 10) - 1 (upper)");
    check_equal_fp(test, y.lower, 3e-18, "exp(x - 10) - 1 (lower)");

    y = t.exp();
    check_equal_fp(test, y.upper, 1.0000000000000038, "exp(t) (upper)");
    check_close_fp(test, y.lower, -2.4663276224724858e-17, 5e-16, "exp(t) (lower)");

    // The actual relative error for exp(709).lower is approx. 2e-14.
    // Can we do better?
    y = DoubleDouble(709.0).exp();
    check_equal_fp(test, y.upper, 8.218407461554972e+307, "exp(709) upper");
    check_close_fp(test, y.lower, -1.955965507696277e+291, 1e-13, "exp(709) lower");

    y = DoubleDouble(710.0).exp();
    check_true(test, std::isinf(y.upper), "isinf(exp(710).upper)");
    check_equal_fp(test, y.lower, 0.0, "exp(710).lower == 0");

    y = one.expm1();
    check_equal_fp(test, y.upper, 1.7182818284590453, "expm1(1) upper");
    check_equal_fp(test, y.lower, -7.747991575210629e-17, "expm1(1) lower");

    y = DoubleDouble(0.46875).expm1();
    check_equal_fp(test, y.upper, 0.5979954499506333, "expm1(0.46875) upper");
    check_equal_fp(test, y.lower, 1.6864630310268093e-17, "expm1(0.46875) lower");

    y = t.expm1();
    check_equal_fp(test, y.upper, 3.7500950075008074e-15, "expm1(t) upper");
    check_equal_fp(test, y.lower, -6.814186434788356e-32, "expm1(t) lower");

    y = DoubleDouble(-0.000244140625).expm1();
    check_equal_fp(test, y.upper, -0.00024411082510278348, "expm1(-2**-12) upper");
    check_equal_fp(test, y.lower, -7.227720384831839e-21, "expm1(-2**-12) lower");

    y = DoubleDouble(-0.46875).expm1();
    check_equal_fp(test, y.upper, -0.37421599039540887, "expm1(-0.46875) upper");
    check_equal_fp(test, y.lower, -7.658883125910196e-18, "expm1(-0.46875) lower");

    y = 2.0_dd;
    check_equal_fp(test, y.upper, 2.0, "Test of 2.0_dd (upper)");
    check_equal_fp(test, y.lower, 0.0, "Test of 2.0_dd (lower)");

    y = -1.2345e-6_dd;
    check_equal_fp(test, y.upper, -1.2345e-6, "Test of -1.2345e-6_dd (upper)");
    check_equal_fp(test, y.lower, -3.412120026781239e-24, "Test of -1.2345e-6_dd (lower)");

    return test.print_summary("Summary: ");
}
