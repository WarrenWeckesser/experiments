The files here demonstrate the use of the GSL ODE solver `odeiv2`
(see https://www.gnu.org/software/gsl/doc/html/ode-initval.html)
to solve the Rössler system of ordinary differential equations.

The main program is in `rossler_solve.c`, and the functions that
implement the system of differential equations and its Jacobian
matrix are in `rossler_system.c`.
