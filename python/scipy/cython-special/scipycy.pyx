#!/usr/bin/env python3
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

from scipy.special.cython_special cimport chdtriv

cpdef float cy_chdtriv_float(float p, float x):
    return chdtriv(p, x)

cpdef double cy_chdtriv_double(double p, double x):
    return chdtriv(p, x)
