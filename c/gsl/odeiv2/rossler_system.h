//
// rossler_system.h
//

#ifndef ROSSLER_SYSTEM_H
#define ROSSLER_SYSTEM_H

int rossler_system(double, const double [], double [], void *);
int rossler_jac(double, const double [], double *, double *, void *);

#endif
