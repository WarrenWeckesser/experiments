/*
 *  chua2_cv7.h
 *
 *  CVODE C prototype file for the functions defined in chua2_cv7.c
 *
 *  This file was generated by the program VFGEN, version: 2.6.0.dev5
 *  Generated on 22-Sep-2024 at 16:56
 */

#ifndef CHUA2_C7_H
#define CHUA2_C7_H

int chua2_vf(sunrealtype, N_Vector, N_Vector, void *);
int chua2_jac(sunrealtype, N_Vector, N_Vector, SUNMatrix, void *,
                 N_Vector, N_Vector, N_Vector);
int chua2_func(sunrealtype, N_Vector, sunrealtype *, void *);

#endif /* CHUA2_C7_H */