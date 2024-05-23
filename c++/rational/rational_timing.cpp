//
// Experiments with evaluating a rational function.
//

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

#define BOOST_MATH_RATIONAL_METHOD 2
#include "boost/math/tools/rational.hpp"

#include "rational.hpp"

using namespace std;

using Clock = std::chrono::steady_clock;
using Second = std::chrono::duration<double, std::ratio<1> >;

#define NUM 200000000

void print_row(string name, double x, double delta)
{
    cout << setw(40) << left << name;
    cout << setw(12) << x;
    cout << setw(24) << delta;
    cout << endl;
}

void print_header()
{
    cout << setw(43) << "x";
    cout << setw(18) << "time (sec)";
    cout << endl;
}

int main()
{
    // These are the same rational function coefficients as used in
    // rational::fixed_rational().
    const vector<double> P{1.0, 0.5, 0.125, -0.125, 0.0, 0.5, 0.0, -0.05};
    const vector<double> Q{1.0, 0.25, 0.375, 0.125, 0.5, 0.5, -0.5, 0.25};

    print_header();

    {
        double x = 1.25;
        auto tstart = Clock::now();
        for (int i = 0; i < NUM; ++i) {
            auto y = rational::fixed_rational_loop(x);
            x = x + y;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        print_row("fixed_rational_loop(x)", x, delta);
    }

    {
        double x = 1.25;
        auto tstart = Clock::now();
        for (int i = 0; i < NUM; ++i) {
            auto y = rational::fixed_rational(x);
            x = x + y;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        print_row("fixed_rational(x)", x, delta);
    }

    {
        double x = 1.25;
        auto tstart = Clock::now();
        for (int i = 0; i < NUM; ++i) {
            auto y = rational::eval_rational(x, P, Q);
            x = x + y;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        print_row("eval_rational(x, P, Q)", x, delta); 
    }

    {
        // These are the same rational function coefficients as used in
        // rational::fixed_rational().
        std::vector<double> U;
        std::vector<double> V;
        U.push_back(1.0);
        U.push_back(0.5);
        U.push_back(0.125);
        U.push_back(-0.125);
        U.push_back(0.0);
        U.push_back(0.5);
        U.push_back(0.0);
        U.push_back(-0.05);
        V.push_back(1.0);
        V.push_back(0.25);
        V.push_back(0.375);
        V.push_back(0.125);
        V.push_back(0.5);
        V.push_back(0.5);
        V.push_back(-0.5);
        V.push_back(0.25);

        double x = 1.25;
        auto tstart = Clock::now();
        for (int i = 0; i < NUM; ++i) {
            auto y = rational::eval_rational(x, U, V);
            x = x + y;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        print_row("eval_rational(x, U, V)", x, delta); 
    }

    {
        double x = 1.25;
        auto tstart = Clock::now();
        for (int i = 0; i < NUM; ++i) {
            auto y = rational::eval_rational_with_inversion(x, P, Q);
            x = x + y;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        print_row("eval_rational_with_inversion(x, P, Q)", x, delta); 
    }

    {
        const double a[8] = {1.0, 0.5, 0.125, -0.125, 0.0, 0.5, 0.0, -0.05};
        const double b[8] = {1.0, 0.25, 0.375, 0.125, 0.5, 0.5, -0.5, 0.25};

        double x = 1.25;
        auto tstart = Clock::now();
        for (int i = 0; i < NUM; ++i) {
            auto y = boost::math::tools::evaluate_rational(a, b, x);
            x = x + y;
        }
        auto delta = std::chrono::duration_cast<Second>(Clock::now() - tstart).count();
        print_row("boost::...::evaluate_rational(a, b, x)", x, delta); 
        std::cout << "   (BOOST_MATH_RATIONAL_METHOD = " << BOOST_MATH_RATIONAL_METHOD << ")" << std::endl;
    }
}
