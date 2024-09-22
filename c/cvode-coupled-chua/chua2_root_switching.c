
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Include headers for CVODE */
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> /* dense SUNLinearSolver */
#include <sunmatrix/sunmatrix_dense.h> /* dense SUNMatrix       */

#include "chua2_cv7.h"


int check_int_status(int retval, char *funcname)
{
    if (retval != 0) {
        fprintf(stderr, "SUNDIALS ERROR: %s() failed - returned %d\n", funcname, retval);
        return 1;
    }
    return 0;
}

int check_pointer(void *ptr, char *funcname)
{
    if (ptr == NULL) {
        fprintf(stderr, "SUNDIALS ERROR: %s() failed - returned NULL\n", funcname);
        return 1;
    }
    return 0;
}

int use(const char *program_name, int nv, char *vname[], double y_[], int np, char *pname[], const double p_[])
{
    int i;
    printf("use: %s [options]\n", program_name);
    printf("options:\n");
    printf("    -h    Print this help message.\n");
    for (i = 0; i < nv; ++i) {
        printf("    %s=<initial_condition>   Default value is %e\n", vname[i], y_[i]);
    }
    for (i = 0; i < np; ++i) {
        printf("    %s=<parameter_value>   Default value is %e\n", pname[i], p_[i]);
    }
    printf("    abserr=<absolute_error_tolerance>\n");
    printf("    relerr=<relative_error_tolerance>\n");
    printf("    stoptime=<stop_time>\n");
    return 0;
}


int assign(char *str[], int ns, double v[], char *a)
{
    int i;
    char name[256];
    char *e;

    e = strchr(a, '=');
    if (e == NULL) {
        return -1;
    }
    *e = '\0';
    strcpy(name, a);
    *e = '=';
    ++e;
    for (i = 0; i < ns; ++i) {
        if (strcmp(str[i], name) == 0) {
            break;
        }
    }
    if (i == ns) {
        return -1;
    }
    v[i] = atof(e);
    return i;
}


static void PrintRootInfo(int rootsfound[4])
{
    fprintf(stderr, "rootsfound:");
    for (int k = 0; k < 4; ++k) {
        fprintf(stderr, " %2d", rootsfound[k]);
    }
    fprintf(stderr, "\n");
}

int main (int argc, char *argv[])
{
    SUNContext sunctx;
    int i, j;
    int retval;
    const int N_ = 6;
    const int P_ = 9;
    const sunrealtype def_p_[9] = {
        SUN_RCONST(10.0),   // alpha   
        SUN_RCONST(14.87),  // beta
        SUN_RCONST(-1.27),  // a
        SUN_RCONST(-0.68),  // b
        SUN_RCONST(6.0),    // delta_x
        SUN_RCONST(0.0),    // delta_y
        SUN_RCONST(0.0),    // delat_z
        SUN_RCONST(0.0),    // Q
        SUN_RCONST(0.0)     // Qp
    };
    int rootsfound[4];


    /* Create the SUNDIALS context */
    retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (check_int_status(retval, "SUNContext_Create")) {
        return 1;
    }

    sunrealtype def_y_[6] = {SUN_RCONST(-0.5),
                             SUN_RCONST(0.125),
                             SUN_RCONST(0.0),
                             SUN_RCONST(-0.5),
                             SUN_RCONST(0.25),
                             SUN_RCONST(0.0)};
    sunrealtype y_[6];
    sunrealtype p_[9];
    sunrealtype solver_param_[3] = {SUN_RCONST(1.0e-12), SUN_RCONST(1e-9), SUN_RCONST(10.0)};
    char *varnames_[6] = {"x", "y", "z", "xp", "yp", "zp"};
    char *parnames_[9] = {"alpha", "beta", "a", "b", "delta_x", "delta_y", "delta_z", "Q", "Qp"};
    char *solver_param_names_[3] = {"abserr", "relerr", "stoptime"};

    for (i = 0; i < N_; ++i) {
        y_[i] = def_y_[i];
    }
    for (i = 0; i < P_; ++i) {
        p_[i] = def_p_[i];
    }
    for (i = 1; i < argc; ++i) {
        int j;
        if (strcmp(argv[i], "-h") == 0) {
            use(argv[0], N_, varnames_, def_y_, P_, parnames_, def_p_);
            exit(0);
        }
        j = assign(varnames_, N_, y_, argv[i]);
        if (j == -1) {
            j = assign(parnames_, P_, p_, argv[i]);
            if (j == -1) {
                j = assign(solver_param_names_, 3, solver_param_, argv[i]);
                if (j == -1) {
                    fprintf(stderr, "unknown argument: %s\n", argv[i]);
                    use(argv[0], N_, varnames_, def_y_, P_, parnames_, def_p_); 
                    exit(-1);
                }
            }
        }
    }

    /*
     * Reset Q and Qp to be consistent with x (y_[0]) and xp (y_[3]).
     */
    if (y_[0] < -1) {
        p_[7] = -1;
    }
    else if (y_[0] > 1) {
        p_[7] = 1;
    }
    else {
        p_[7] = 0;
    }
    if (y_[3] < -1) {
        p_[8] = -1;
    }
    else if (y_[3] > 1) {
        p_[8] = 1;
    }
    else {
        p_[8] = 0;
    }

    /* Initial conditions */
    N_Vector y0_ = N_VNew_Serial(N_, sunctx);
    if (check_pointer((void*)y0_, "N_VNew_Serial")) {
        return (1);
    }
    for (i = 0; i < N_; ++i) {
        NV_Ith_S(y0_, i) = y_[i];
    }

    /* Use CV_ADAMS for non-stiff problems, and CV_BDF for stiff problems:   */
    void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (check_pointer((void*)cvode_mem, "CVodeCreate")) {return 1;}

    sunrealtype t = SUN_RCONST(0.0);
    retval = CVodeInit(cvode_mem, chua2_vf, t, y0_);
    if (check_int_status(retval, "CVodeInit")) {
        return 1;
    }
    retval = CVodeSStolerances(cvode_mem, solver_param_[1], solver_param_[0]);
    if (check_int_status(retval, "CVodeSStolerances")) {return 1;}

    /* Call CVodeRootInit to specify the root function g with 2 components */
    retval = CVodeRootInit(cvode_mem, 4, chua2_func);
    if (check_int_status(retval, "CVodeRootInit")) { return 1; }

    retval = CVodeSetUserData(cvode_mem, &(p_[0]));
    if (check_int_status(retval, "CVodeSetUserData()")) {return 1;}

    /* Create dense SUNMatrix for use in linear solves */
    SUNMatrix A = SUNDenseMatrix(N_, N_, sunctx);
    if (check_pointer((void*)A, "SUNDenseMatrix()")) {return 1;}

    /* Create dense SUNLinearSolver object for use by CVode */
    SUNLinearSolver LS = SUNLinSol_Dense(y0_, A, sunctx);
    if (check_pointer((void*)LS, "SUNLinSol_Dense()")) {return 1;}

    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_int_status(retval, "CVodeSetLinearSolver()")) {return 1;}

    /* Set the Jacobian routine */
    retval = CVodeSetJacFn(cvode_mem, chua2_jac);
    if (check_int_status(retval, "CVodeSetJacFn()")) {return 1;}

    sunrealtype t1 = solver_param_[2];
    printf("t, x, y, z, xp, yp, zp, x_equals_neg1, x_equals_pos1, xp_equals_neg1, xp_equals_pos1, Q, Qp\n");
    /* Print the initial condition  */
    printf("%.8e", t);
    for (j = 0; j < N_; ++j) {
        printf(", %.8e", NV_Ith_S(y0_, j));
    }
    sunrealtype funcval[4];
    chua2_func(t, y0_, funcval, (void *) p_);
    printf(", %.8e",funcval[0]);
    printf(", %.8e",funcval[1]);
    printf(", %.8e",funcval[2]);
    printf(", %.8e",funcval[3]);
    printf(", %3f, %3f", p_[7], p_[8]);
    printf("\n");

    retval = CVodeSetStopTime(cvode_mem, t1);
    if (check_int_status(retval, "CVodeSetStopTime()")) {return 1;}

    while (t < t1) {
        /* Advance the solution. */
        retval = CVode(cvode_mem, t1, y0_, &t, CV_ONE_STEP);

        if (retval == CV_ROOT_RETURN) {
            /* Hit x == -1, x == 1, xp == -1 or xp == 1. */
            int retvalr = CVodeGetRootInfo(cvode_mem, rootsfound);
            if (check_int_status(retvalr, "CVodeGetRootInfo")) { return 1; }
            PrintRootInfo(rootsfound);
            fprintf(stderr, "t = %13.8f  x = %13.8f  xp = %13.8f\n", t, NV_Ith_S(y0_, 0), NV_Ith_S(y0_, 3));

            if (rootsfound[0] == -1) {
                p_[7] = -1;
            }
            else if (rootsfound[0] == 1) {
                p_[7] = 0;
            }
            else if (rootsfound[1] == -1) {
                p_[7] = 0;
            }
            else if (rootsfound[1] == 1) {
                p_[7] = 1;
            }

            if (rootsfound[2] == -1) {
                p_[8] = -1;
            }
            else if (rootsfound[2] == 1) {
                p_[8] = 0;
            }
            else if (rootsfound[3] == -1) {
                p_[8] = 0;
            }
            else if (rootsfound[3] == 1) {
                p_[8] = 1;
            }
            int reinit_status = CVodeReInit(cvode_mem, t, y0_);
            if (check_int_status(reinit_status, "CVodeReInit")) { return 1; }

            continue;
        }

        if (retval != CV_SUCCESS && retval != CV_TSTOP_RETURN) {
            fprintf(stderr, "retval=%d\n", retval);
            retval = -1;
            break;
        }
        else {
            retval = 0;
        }
        /* Print the solution at the current time */
        printf("%.8e", t);
        for (j = 0; j < N_; ++j) {
            printf(", %.8e", NV_Ith_S(y0_,j));
        }
        chua2_func(t, y0_, funcval, (void *) p_);
        printf(", %.8e", funcval[0]);
        printf(", %.8e", funcval[1]);
        printf(", %.8e", funcval[2]);
        printf(", %.8e", funcval[3]);
        printf(", %3f, %3f", p_[7], p_[8]);
        printf("\n");
    }

    /* Free memory */
    N_VDestroy(y0_);
    CVodeFree(&cvode_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    SUNContext_Free(&sunctx);
    return retval;
}
