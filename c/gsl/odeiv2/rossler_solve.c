//
//  rossler_solve.c
//

#include <string.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include "rossler_system.h"


int main (int argc, char *argv[])
{
    const int N = 3;

    // Rossler parameters
    double a = 0.17;
    double b = 0.4;
    double c = 8.5;

    double params[] = {a, b, c};

    double abserr = 1.0e-12;
    double relerr = 1.0e-9;

    double stoptime = 10.0;

    // Initial conditions
    double y[] = {0.01, 0.0, 0.0};

    const gsl_odeiv2_step_type *T  = gsl_odeiv2_step_rk8pd;
    gsl_odeiv2_step    *step    = gsl_odeiv2_step_alloc(T, N);
    gsl_odeiv2_control *control = gsl_odeiv2_control_y_new(abserr, relerr);
    gsl_odeiv2_evolve  *evolve  = gsl_odeiv2_evolve_alloc(N);
    gsl_odeiv2_system sys = {rossler_system, rossler_jac, N, params};

    double t = 0.0;
    double t1 = stoptime;
    double h = 1e-6;

    while (t < t1) {
        int status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, &t, t1, &h, y);
        if (status != GSL_SUCCESS) {
            fprintf(stderr, "status=%d\n", status);
            break;
        }
        printf("%.8e", t);
        for (int j = 0; j < N; ++j) {
            printf(" %.8e", y[j]);
        }
        printf("\n");
    }

    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(step);

    return 0;
}
