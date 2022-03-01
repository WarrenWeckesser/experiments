#include <cstdio>
#include <vector>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

typedef boost::numeric::ublas::vector<double> state_type;


class logistic
{
    double r;  // Small population growth rate
    double K;  // Carrying capacity

public:

    logistic(double r, double K) : r(r), K(K) {}

    void operator()(const state_type &y, state_type &dydt, const double /* t */)
    {
        dydt[0] = r*y[0]*(1 - y[0]/K);
    }
};


void writer(const state_type &y, const double t)
{
    printf("%7.1f ", t);
    for (auto v : y) {
        printf(" %10.7f", v);
    }
    printf("\n");
}


int main(int argc, char *argv[])
{
    state_type y(1);
    double initial_step = 1e-6;
    vector<double> times = {0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
                            64.0, 128.0, 256.0, 512.0};

    // Initial condition
    y[0] = 1e-4;

    bulirsch_stoer_dense_out<state_type> stepper(1e-9, 1e-9, 1.0, 0.0);
    auto sys = logistic(0.08, 3.0);
    integrate_times(stepper, sys, y,
                    times.begin(), times.end(),
                    initial_step, writer);
}
