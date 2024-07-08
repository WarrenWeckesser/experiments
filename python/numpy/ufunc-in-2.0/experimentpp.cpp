/* -*- c -*- */

/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/npy_math.h"



template<typename T>
static int
deadzone_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                      char **args, npy_intp const *dimensions,
                      npy_intp const *steps, NpyAuxData *NPY_UNUSED(func))
{
    char *p_x = args[0];
    char *p_low = args[1];
    char *p_high = args[2];
    char *p_out = args[3];
    npy_intp n = dimensions[0];
    npy_intp x_stride = steps[0];
    npy_intp low_stride = steps[1];
    npy_intp high_stride = steps[2];
    npy_intp out_stride = steps[3];

    for (npy_intp i = 0; i < n; i++,
                                p_x += x_stride,
                                p_low += low_stride,
                                p_high += high_stride,
                                p_out += out_stride) {
        T x = *(T *) p_x;
        T low = *(T *) p_low;
        T high = *(T *) p_high;
        T result;

        if (x < low) {
            result = x - low;
        }
        else if (x > high) {
            result = x - high;
        }
        else {
            result = 0.0;
        }
        *(T *) p_out = result;
    }
    return 0;
}

extern "C" {

static int
deadzone_float_strided_loop(PyArrayMethod_Context *context,
                            char **args, npy_intp const *dimensions,
                            npy_intp const *steps, NpyAuxData *func)
{
    return deadzone_strided_loop<float>(context, args, dimensions, steps, func);
}

static int
deadzone_double_strided_loop(PyArrayMethod_Context *context,
                             char **args, npy_intp const *dimensions,
                             npy_intp const *steps, NpyAuxData *func)
{
    return deadzone_strided_loop<double>(context, args, dimensions, steps, func);
}

}  // extern "C"

static int
add_deadzone(PyObject *module, PyObject *dict) {
    PyObject* deadzone = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 3, 1,
                                                 PyUFunc_None, "deadzone", NULL, 0);
    if (deadzone == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[] = {
        &PyArray_DoubleDType, &PyArray_DoubleDType, &PyArray_DoubleDType, &PyArray_DoubleDType
    };

    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, (void *) deadzone_double_strided_loop},
        {0, NULL}
    };

    PyArrayMethod_Spec spec = {
        .name = "deadzone_double_strided_loop",
        .nin = 3,
        .nout = 1,
        .casting = NPY_UNSAFE_CASTING,
        .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
        .dtypes = dtypes,
        .slots = slots
    };

    if (PyUFunc_AddLoopFromSpec(deadzone, &spec) < 0) {
        Py_DECREF(deadzone);
        return -1;
    }

/*
    {
        PyArray_DTypeMeta *dtypes[] = {
            &PyArray_FloatDType, &PyArray_FloatDType, &PyArray_FloatDType, &PyArray_FloatDType
        };

        PyType_Slot slots[] = {
            {NPY_METH_strided_loop, (void *) deadzone_float_strided_loop},
            {0, NULL}
        };

        PyArrayMethod_Spec spec = {
            .name = "deadzone_float_strided_loop",
            .nin = 3,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
            .dtypes = dtypes,
            .slots = slots
        };

        if (PyUFunc_AddLoopFromSpec(deadzone, &spec) < 0) {
            Py_DECREF(deadzone);
            return -1;
        }
    }
*/
 
    PyDict_SetItemString(dict, "deadzone", deadzone);
    Py_DECREF(deadzone);
    return 0;
}

static PyMethodDef ExperimentPPMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "experimentpp",
        NULL,
        -1,
        ExperimentPPMethods,
        NULL,
        NULL,
        NULL,
        NULL
};


extern "C" {

PyMODINIT_FUNC PyInit_experimentpp(void) {
    PyObject *module;
    PyObject *module_dict;

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

    module_dict = PyModule_GetDict(module);

    if (add_deadzone(module, module_dict) < 0) {
        Py_DECREF(module);
        PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load exeriment module.");
        return NULL;
    }
    return module;
}

}  // extern "C"
