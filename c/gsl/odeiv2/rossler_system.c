//
//  rossler_gvf.c
//

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>


int rossler_system(double t, const double w[], double f[], void *params)
{
    double x, y, z;
    double a, b, c;

    x = w[0];
    y = w[1];
    z = w[2];

    a = ((double *) params)[0];
    b = ((double *) params)[1];
    c = ((double *) params)[2];

    f[0] = -y - z;
    f[1] = x + y*a;
    f[2] = b + (x - c)*z;

    return GSL_SUCCESS;
}

//
//  The Jacobian matrix.
//

int rossler_jac(double t, const double w[], double *jac, double *dfdt, void *params)
{
    double x, y, z;
    double a, b, c;

    x = w[0];
    y = w[1];
    z = w[2];

    a = ((double *) params)[0];
    b = ((double *) params)[1];
    c = ((double *) params)[2];

    gsl_matrix_view jac_mat = gsl_matrix_view_array(jac, 3, 3);
    gsl_matrix *m = &jac_mat.matrix;

    gsl_matrix_set(m, 0, 0, 0.0);
    gsl_matrix_set(m, 0, 1, -1.0);
    gsl_matrix_set(m, 0, 2, -1.0);
    gsl_matrix_set(m, 1, 0, 1.0);
    gsl_matrix_set(m, 1, 1, a);
    gsl_matrix_set(m, 1, 2, 0.0);
    gsl_matrix_set(m, 2, 0, z);
    gsl_matrix_set(m, 2, 1, 0.0);
    gsl_matrix_set(m, 2, 2, x - c);

    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    dfdt[2] = 0.0;

    return GSL_SUCCESS;
}
