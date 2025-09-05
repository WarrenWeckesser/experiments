
#include <iostream>
#include <ginac/ginac.h>
#include "signcases.h"


static GiNaC::ex
signcases_eval(const GiNaC::ex &cond,
               const GiNaC::ex &ltzero,
               const GiNaC::ex &eqzero,
               const GiNaC::ex &gtzero)
{
    return signcases(cond, ltzero, eqzero, gtzero).hold();
}

static GiNaC::ex
signcases_evalf(const GiNaC::ex &cond,
                const GiNaC::ex &ltzero,
                const GiNaC::ex &eqzero,
                const GiNaC::ex &gtzero)
{
    return signcases(cond, ltzero, eqzero, gtzero).hold();
}

/*
static GiNaC::ex
signcases_deriv(const GiNaC::ex &cond,
                const GiNaC::ex &ltzero,
                const GiNaC::ex &eqzero,
                const GiNaC::ex &gtzero,
                unsigned diff_param)
{
    return signcases(cond, ltzero, eqzero, gtzero).hold();
}
*/

// f(x)  = signcases(x, x, 0, x**2)
// f'(x) = signcases(x, 1, 0, 2*x)
// f(x)  = signcases(c(x), lt(x), eq(x), gt(x))
// What I get
//   f'(x) = sc_0()*c'(x) + sc_1()*lt'(x) + sc_2()*eq'(x) + sc_3()*gt'(x)
// What I want
//   f'(x) = signcases(c(x), lt'(x), eq'(x), gt'(x))

REGISTER_FUNCTION(signcases,
    eval_func(signcases_eval).
    evalf_func(signcases_evalf));



static GiNaC::ex
step1_eval(const GiNaC::ex &x)
{
    return step1(x).hold();
}

static GiNaC::ex
step1_evalf(const GiNaC::ex &x)
{
    return step1(x).hold();
}


REGISTER_FUNCTION(step1,
    eval_func(step1_eval).
    evalf_func(step1_evalf));

int
main(int argc, char *argv[])
{
    GiNaC::symbol x("x"), y("y");
    GiNaC::ex e, der;

    //e = signcases(x-y, x+y*y, sin(x)+y, y-3*x);
    //e = step1(x*x-y*y);
    e = GiNaC::step(x*x-y*y);
    der = e.diff(x);
    std::cout << e << std::endl;
    std::cout << der << std::endl;
    std::cout << GiNaC::step(-1) << " " << GiNaC::step(0) << " " << GiNaC::step(1) << std::endl;

    return 0;
}