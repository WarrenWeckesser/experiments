
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Define the gufunc 'euclidean_pdist'
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int euclidean_pdist_process_core_dims(PyUFuncObject *ufunc,
                                      npy_intp *core_dim_sizes)
{
    npy_intp m_even, m_odd;
    npy_intp required_p;

    //
    // core_dim_sizes will hold the core dimensions [m, n, p].
    // p will be -1 if the caller did not provide the out argument.
    //
    npy_intp m = core_dim_sizes[0];
    npy_intp p = core_dim_sizes[2];
    if (m == 0) {
        PyErr_SetString(PyExc_ValueError,
                "euclidean_pdist: the core dimension m of the input parameter "
                "x must be at least 1; got 0.");
        return -1;
    }

    // Identify the even and odd factors of the expression m*(m - 1).
    npy_intp r = m % 2;
    m_even = m - r;
    m_odd = m - (1 - r);

    if (m_even/2 > NPY_MAX_INTP/m_odd) {
        PyErr_Format(PyExc_ValueError,
                "euclidean_pdist: the core dimension m=%zd of the input "
                "parameter x is too big; the calculation of the output size "
                "p = m*(m - 1)/2 results in integer overflow.", m);
        return -1;
    }

    required_p = m_odd*(m_even/2);

    if (p == -1) {
        core_dim_sizes[2] = required_p;
        return 0;
    }
    if (p != required_p) {
        PyErr_Format(PyExc_ValueError,
                "euclidean_pdist: the core dimension p of the out parameter "
                "does not equal m*(m - 1)/2, where m is the first core "
                "dimension of the input x; got m=%zd, so p must be %zd, "
                "but got p=%zd).",
                m, required_p, p);
        return -1;
    }
    return 0;
}

static void
euclidean_pdist_double_loop(char **args,
                            npy_intp const *dimensions,
                            npy_intp const *steps,
                            void *NPY_UNUSED(func))
{
    // Input and output arrays
    char *p_x = args[0];
    char *p_out = args[1];
    // Number of loops of pdist calculations to execute.
    npy_intp nloops = dimensions[0];
    // Core dimensions
    npy_intp m = dimensions[1];
    npy_intp n = dimensions[2];
    // npy_intp p = dimensions[3]; // Unused; we know it must be m*(m-1)/2.
    // Core strides
    npy_intp x_stride = steps[0];
    npy_intp out_stride = steps[1];
    // x array strides
    npy_intp x_row_stride = steps[2];
    npy_intp x_col_stride = steps[3];
    // out array strides
    npy_intp out_inner_stride = steps[4];

    for (npy_intp loop = 0; loop < nloops; ++loop, p_x += x_stride,
                                                   p_out += out_stride) {
        npy_intp k_out = 0;
        for (npy_intp i = 0; i < m - 1; ++i) {
            char *p_rowi = p_x + i*x_row_stride;
            for (npy_intp j = i + 1; j < m; ++j) {
                char *p_rowj = p_x + j*x_row_stride;
                double sum = 0.0;
                for (npy_intp k = 0; k < n; ++k) {
                    double x1 = *(double *)(p_rowi + k*x_col_stride);
                    double x2 = *(double *)(p_rowj + k*x_col_stride);
                    double delta = x1 - x2;
                    sum += delta * delta;
                }
                double norm = sqrt(sum);
                *(double *)(p_out + k_out*out_inner_stride) = norm;
                ++k_out;
            }
        }
    }
}


static PyUFuncGenericFunction euclidean_pdist_functions[] = {
    (PyUFuncGenericFunction) &euclidean_pdist_double_loop
};
static void *const euclidean_pdist_data[] = {NULL};
static const char euclidean_pdist_typecodes[] = {NPY_DOUBLE, NPY_DOUBLE};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Define the gufunc 'conv1d_full'
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int conv1d_full_process_core_dims(PyUFuncObject *ufunc,
                                  npy_intp *core_dim_sizes)
{
    //
    // core_dim_sizes will hold the core dimensions [m, n, p].
    // p will be -1 if the caller did not provide the out argument.
    //
    npy_intp m = core_dim_sizes[0];
    npy_intp n = core_dim_sizes[1];
    npy_intp p = core_dim_sizes[2];
    npy_intp required_p = m + n - 1;

    if (p == -1) {
        core_dim_sizes[2] = required_p;
        return 0;
    }
    if (p != required_p) {
        PyErr_Format(PyExc_ValueError,
                "conv1d_full: the core dimension p of the out parameter "
                "does not equal m + n - 1, where m and n are the core "
                "dimensions of the inputs x and y; got m=%zd and n=%zd so "
                "p must be %zd, but got p=%zd.",
                m, n, required_p, p);
        return -1;
    }
    return 0;
}

static void
conv1d_full_double_loop(char **args,
                        npy_intp const *dimensions,
                        npy_intp const *steps,
                        void *NPY_UNUSED(func))
{
    // Input and output arrays
    char *p_x = args[0];
    char *p_y = args[1];
    char *p_out = args[2];
    // Number of core calculations to execute.
    npy_intp nloops = dimensions[0];
    // Core dimensions
    npy_intp m = dimensions[1];
    npy_intp n = dimensions[2];
    npy_intp p = dimensions[3]; // Must be m + n - 1.
    // Core strides
    npy_intp x_stride = steps[0];
    npy_intp y_stride = steps[1];
    npy_intp out_stride = steps[2];
    // Inner strides
    npy_intp x_inner_stride = steps[3];
    npy_intp y_inner_stride = steps[4];
    npy_intp out_inner_stride = steps[5];

    for (npy_intp loop = 0; loop < nloops; ++loop, p_x += x_stride,
                                                   p_y += y_stride,
                                                   p_out += out_stride) {
        // Basic implementation of 1d convolution
        for (npy_intp k = 0; k < p; ++k) {
            double sum = 0.0;
            for (npy_intp i = MAX(0, k - n + 1); i < MIN(m, k + 1); ++i) {
                double x_i = *(double *)(p_x + i*x_inner_stride);
                double y_k_minus_i = *(double *)(p_y + (k - i)*y_inner_stride);
                sum +=  x_i * y_k_minus_i;
            }
            *(double *)(p_out + k*out_inner_stride) = sum;
        }
    }
}


static PyUFuncGenericFunction conv1d_full_functions[] = {
    (PyUFuncGenericFunction) &conv1d_full_double_loop
};
static void *const conv1d_full_data[] = {NULL};
static const char conv1d_full_typecodes[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Define the gufunc 'all_equal'
// Shape signature is (m?),(n?)->().
// Requires m == n, or m == 1, or n == 1, to allow broadcasting within
// the core dimensions (implemented in the loop function).
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int all_equal_process_core_dims(PyUFuncObject *ufunc,
                                npy_intp *core_dim_sizes)
{
    npy_intp m = core_dim_sizes[0];
    npy_intp n = core_dim_sizes[1];

    if (!(m == n || m == 1 || n == 1 )) {
        PyErr_Format(PyExc_ValueError,
                "all_equal: the core dimensions m and n must satisfy "
                "m == n, or m == 1, or n == 1.  Got m = %zd, n = %zd.",
                m, n);
        return -1;
    }
    return 0;
}

static void
all_equal_double_loop(char **args,
                      npy_intp const *dimensions,
                      npy_intp const *steps,
                      void *NPY_UNUSED(func))
{
    // Input and output arrays
    char *p_x = args[0];
    char *p_y = args[1];
    char *p_out = args[2];
    // Number of core calculations to execute.
    npy_intp nloops = dimensions[0];
    // Core dimensions
    npy_intp m = dimensions[1];
    npy_intp n = dimensions[2];
    // Core strides
    npy_intp x_stride = steps[0];
    npy_intp y_stride = steps[1];
    npy_intp out_stride = steps[2];
    // Inner strides
    npy_intp x_inner_stride = steps[3];
    npy_intp y_inner_stride = steps[4];

    if (m == 0 || n == 0) {
        // Trivially true.
        *(npy_bool *)p_out = 1;
        return;
    }

    for (npy_intp loop = 0; loop < nloops; ++loop, p_x += x_stride,
                                                   p_y += y_stride,
                                                   p_out += out_stride) {
        npy_bool result = 1;
        for (npy_intp i = 0; i < MAX(m, n); ++i) {
            double x_i = *(double *)(p_x + i*x_inner_stride);
            double y_i = *(double *)(p_y + i*y_inner_stride);
            if (x_i != y_i) {
                result = 0;
                break;
            }
        }
        *(npy_bool *)p_out = result;
    }
}


static PyUFuncGenericFunction all_equal_functions[] = {
    (PyUFuncGenericFunction) &all_equal_double_loop
};
static void *const all_equal_data[] = {NULL};
static const char all_equal_typecodes[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_BOOL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Define the gufunc 'cross'
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int cross_process_core_dims(PyUFuncObject *ufunc,
                            npy_intp *core_dim_sizes)
{
    //
    // core_dim_sizes will hold the core dimensions [m, p].
    // p will be -1 if the caller did not provide the out argument.
    //
    npy_intp m = core_dim_sizes[0];
    npy_intp p = core_dim_sizes[1];

    if (m != 2 && m != 3) {
        PyErr_Format(PyExc_ValueError,
                "cross: core dimension of the input must be 2 or 3; "
                "got %zd.", m);
    }

    npy_intp required_p = (m == 2) ? 1 : 3;

    if (p == -1) {
        core_dim_sizes[1] = required_p;
        return 0;
    }
    if (p != required_p) {
        PyErr_Format(PyExc_ValueError,
                "cross: the core dimension p of the out parameter "
                "must be 1 or 3 if the input core dimension is 2 or 3, "
                "respectively. Got input core dimension %zd, but p is %zd",
                m, p);
        return -1;
    }
    return 0;
}

static void
cross_double_loop(char **args,
                  npy_intp const *dimensions,
                  npy_intp const *steps,
                  void *NPY_UNUSED(func))
{
    // Input and output arrays
    char *p_x = args[0];
    char *p_y = args[1];
    char *p_out = args[2];
    // Number of core calculations to execute.
    npy_intp nloops = dimensions[0];
    // Core dimensions
    npy_intp m = dimensions[1];
    // npy_intp p = dimensions[2];  // Unused here; will be 1 or 3, depending on m.
    // Core strides
    npy_intp x_stride = steps[0];
    npy_intp y_stride = steps[1];
    npy_intp out_stride = steps[2];
    // Inner strides
    npy_intp x_inner_stride = steps[3];
    npy_intp y_inner_stride = steps[4];
    npy_intp out_inner_stride = steps[5];

    if (m == 3) {
        for (npy_intp loop = 0; loop < nloops; ++loop, p_x += x_stride,
                                                       p_y += y_stride,
                                                       p_out += out_stride) {
            double x0 = *(double *)(p_x);
            double x1 = *(double *)(p_x + x_inner_stride);
            double x2 = *(double *)(p_x + 2*x_inner_stride);
            double y0 = *(double *)(p_y);
            double y1 = *(double *)(p_y + y_inner_stride);
            double y2 = *(double *)(p_y + 2*y_inner_stride);
            *(double *)(p_out)                      = x1*y2 - x2*y1;
            *(double *)(p_out + out_inner_stride)   = x2*y0 - x0*y2;
            *(double *)(p_out + 2*out_inner_stride) = x0*y1 - x1*y0;
        }
    }
    else {
        // m == 2
        for (npy_intp loop = 0; loop < nloops; ++loop, p_x += x_stride,
                                                       p_y += y_stride,
                                                       p_out += out_stride) {
            double x0 = *(double *)(p_x);
            double x1 = *(double *)(p_x + x_inner_stride);
            double y0 = *(double *)(p_y);
            double y1 = *(double *)(p_y + y_inner_stride);
            *(double *)p_out = x0*y1 - x1*y0;
        }
    }
}


static PyUFuncGenericFunction cross_functions[] = {
    (PyUFuncGenericFunction) &cross_double_loop
};
static void *const cross_data[] = {NULL};
static const char cross_typecodes[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};



// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Extension module boilerplate...
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef experiment_methods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "experiment",
        "experiment module",
        -1,
        experiment_methods,
};


PyMODINIT_FUNC PyInit_experiment(void)
{
    PyObject *module;
    PyUFuncObject *gufunc;
    int status;

    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;
    }

    if (PyArray_ImportNumPyAPI() < 0) {
        Py_DECREF(module);
        return NULL;
    }
    if (PyUFunc_ImportUFuncAPI() < 0) {
        Py_DECREF(module);
        return NULL;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Define the gufunc 'euclidean_pdist'
    // Shape signature is (m,n)->(p) where p must be m*(m-1)/2.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    gufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                                euclidean_pdist_functions,
                                euclidean_pdist_data,
                                euclidean_pdist_typecodes,
                                1, 1, 1, PyUFunc_None, "euclidean_pdist",
                                "pairwise euclidean distance of rows in x",
                                0, "(m,n?)->(p)");
    if (gufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    gufunc->process_core_dims_func = &euclidean_pdist_process_core_dims;

    status = PyModule_AddObject(module, "euclidean_pdist",
                                (PyObject *) gufunc);
    if (status == -1) {
        Py_DECREF(gufunc);
        Py_DECREF(module);
        return NULL;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Define the gufunc 'conv1d_full'
    // Shape signature is (m),(n)->(p) where p must be m + n - 1.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    gufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                                conv1d_full_functions,
                                conv1d_full_data,
                                conv1d_full_typecodes,
                                1, 2, 1, PyUFunc_None, "conv1d_full",
                                "convolution of x and y ('full' mode)",
                                0, "(m),(n)->(p)");
    if (gufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    gufunc->process_core_dims_func = &conv1d_full_process_core_dims;

    status = PyModule_AddObject(module, "conv1d_full",
                                (PyObject *) gufunc);
    if (status == -1) {
        Py_DECREF(gufunc);
        Py_DECREF(module);
        return NULL;
    }


    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Define the gufunc 'all_equal'
    // Shape signature is (m?),(n?)->().
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    gufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                                all_equal_functions,
                                all_equal_data,
                                all_equal_typecodes,
                                1, 2, 1, PyUFunc_None, "all_equal",
                                "Return true if x1[i] == x2[i] for all i.\n"
                                "\n"
                                "Functionally equivalent to `np.logical_and.reduce(np.equal(x1, x2), axis=-1)`.\n",
                                0, "(m?),(n?)->()");
    if (gufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    gufunc->process_core_dims_func = &all_equal_process_core_dims;

    status = PyModule_AddObject(module, "all_equal",
                                (PyObject *) gufunc);
    if (status == -1) {
        Py_DECREF(gufunc);
        Py_DECREF(module);
        return NULL;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Define the gufunc 'cross'
    // Shape signature is (m),(m)->(p) where m must be 2 or 3, and p must
    // be 0 if m is 2 or 3 if m is 3.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    gufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                                cross_functions,
                                cross_data,
                                cross_typecodes,
                                1, 2, 1, PyUFunc_None, "cross",
                                "cross product (2-d or 3-d)",
                                0, "(m),(m)->(p)");
    if (gufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    gufunc->process_core_dims_func = &cross_process_core_dims;

    status = PyModule_AddObject(module, "cross",
                                (PyObject *) gufunc);
    if (status == -1) {
        Py_DECREF(gufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
