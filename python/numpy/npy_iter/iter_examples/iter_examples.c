//
// Example of wrapping some functions from math.h using the NumPy C API.
//
// The code for cos_v1 was copied from
//   https://lectures.scientific-python.org/advanced/interfacing_with_c/interfacing_with_c.html#numpy-support
// and then modified.
//

#define MODULE_DOCSTRING ""\
"The extension module iter_examples defines these functions:\n"\
"\n"\
"cos_v1(x)\n"\
"    This function was called cos_func_np in the code from\n"\
"    lectores.scientific-python.org.  The code demonstrates the use of\n"\
"    NpyIter_MultiNew.\n"\
"    The argument to cos_v1 must be a numpy array of `np.float64`.  The\n"\
"    code does not show how to handle numpy arrays with different dtypes,\n"\
"    nor how to handle an array-like input (e.g. a list of integers).\n"\
"\n"\
"cos_v2(x)\n"\
"    This function is similar to cos_v1, but the iterator flags\n"\
"    NPY_ITER_EXTERNAL_LOOP, NPY_ITER_BUFFERED and NPY_ITER_GROWINNER\n"\
"    are not set.  This means the iterator will iterate over every\n"\
"    individual element of the input array.\n"\
"\n"\
"hypot_v1(x, y)\n"\
"    A wrapper of the standard libary hypot function.  This code\n"\
"    demonstrates how the iterator is used to implement broadcasting\n"\
"    of two parameters.\n"\
"\n"\
"hypot_v2(x, y)\n"\
"    A wrapper of the standard libary hypot function.  This does the\n"\
"    same calculation as hypot_v1, but the iterator does not set the flags\n"\
"    NPY_ITER_EXTERNAL_LOOP, NPY_ITER_BUFFERED and NPY_ITER_GROWINNER.\n"\
"\n"


#include <Python.h>


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include <stdio.h>

static PyObject* cos_v1(PyObject* self, PyObject* args)
{
    PyArrayObject *arrays[2];  // holds input and output array
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[2];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[2];

    NpyIter_IterNextFunc *iternext;

    // Parse single NumPy array argument
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrays[0])) {
        return NULL;
    }

    arrays[1] = NULL;  // The result will be allocated by the iterator

    // Set up and create the iterator
    iterator_flags = (NPY_ITER_ZEROSIZE_OK |
                      //
                      // Enable buffering in case the input is not behaved
                      // (native byte order or not aligned),
                      // disabling may speed up some cases when it is known to
                      // be unnecessary.
                      //
                      //NPY_ITER_BUFFERED |
                      // Manually handle innermost iteration for speed:
                      NPY_ITER_EXTERNAL_LOOP |
                      NPY_ITER_GROWINNER |
                      0);

    op_flags[0] = (NPY_ITER_READONLY |
                   //
                   // Required that the arrays are well behaved, since the cos
                   // call below requires this.
                   //
                   NPY_ITER_NBO |
                   NPY_ITER_ALIGNED);

    // Ask the iterator to allocate an array to write the output to
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    //
    // Ensure the iteration has the correct type, could be checked
    // specifically here.
    //
    op_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    op_dtypes[1] = op_dtypes[0];

    // Create the NumPy iterator object:
    iter = NpyIter_MultiNew(2, arrays, iterator_flags,
                            // Use input order for output and iteration
                            NPY_KEEPORDER,
                            // Allow only byte-swapping of input
                            NPY_EQUIV_CASTING, op_flags, op_dtypes);
    Py_DECREF(op_dtypes[0]);  // The second one is identical.

    if (iter == NULL)
        return NULL;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // Fetch the output array which was allocated by the iterator:
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);

    if (NpyIter_GetIterSize(iter) == 0) {
        //
        // If there are no elements, the loop cannot be iterated.
        // This check is necessary with NPY_ITER_ZEROSIZE_OK.
        //
        NpyIter_Deallocate(iter);
        return ret;
    }

    // The location of the data pointer which the iterator may update
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    // The location of the stride which the iterator may update
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    // The location of the inner loop size which the iterator may update
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // Iterate over the arrays
    do {
        npy_intp stride = strideptr[0];
        npy_intp count = *innersizeptr;
#ifdef DEBUG
        printf("cos_v1: stride = %zd,  count = %zd\n", stride, count);
#endif
        // out is always contiguous, so use double
        double *out = (double *)dataptr[1];
        char *in = dataptr[0];

        // The output is allocated and guaranteed contiguous (out++ works):
        assert(strideptr[1] == sizeof(double));

        //
        // For optimization it can make sense to add a check for
        // stride == sizeof(double) to allow the compiler to optimize for that.
        //
        while (count--) {
            *out = cos(*(double *)in);
            out++;
            in += stride;
        }
    } while (iternext(iter));

    // Clean up and return the result
    NpyIter_Deallocate(iter);
    return ret;
}


static PyObject* cos_v2(PyObject* self, PyObject* args)
{
    PyArrayObject *arrays[2];  // holds input and output array
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[2];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[2];

    NpyIter_IterNextFunc *iternext;

    // Parse single NumPy array argument
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrays[0])) {
        return NULL;
    }

    arrays[1] = NULL;  // The result will be allocated by the iterator

    iterator_flags = NPY_ITER_ZEROSIZE_OK;

    op_flags[0] = (NPY_ITER_READONLY |
                   //
                   // Required that the arrays are well behaved, since the cos
                   // call below requires this.
                   //
                   NPY_ITER_NBO |
                   NPY_ITER_ALIGNED);

    // Ask the iterator to allocate an array to write the output to
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    //
    // Ensure the iteration has the correct type, could be checked
    // specifically here.
    //
    op_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    op_dtypes[1] = op_dtypes[0];

    // Create the NumPy iterator object:
    iter = NpyIter_MultiNew(2, arrays, iterator_flags,
                            // Use input order for output and iteration
                            NPY_KEEPORDER,
                            // Allow only byte-swapping of input
                            NPY_EQUIV_CASTING, op_flags, op_dtypes);
    Py_DECREF(op_dtypes[0]);  // The second one is identical.

    if (iter == NULL)
        return NULL;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // Fetch the output array which was allocated by the iterator:
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);

    if (NpyIter_GetIterSize(iter) == 0) {
        //
        // If there are no elements, the loop cannot be iterated.
        // This check is necessary with NPY_ITER_ZEROSIZE_OK.
        //
        NpyIter_Deallocate(iter);
        return ret;
    }

    // The location of the data pointer which the iterator may update
    char **dataptr = NpyIter_GetDataPtrArray(iter);

    // Iterate over the array
    do {
        double *out = (double *)dataptr[1];
        double *in = (double *)dataptr[0];
        *out = cos(*in);
    } while (iternext(iter));

    // Clean up and return the result
    NpyIter_Deallocate(iter);
    return ret;
}

static PyObject* hypot_v1(PyObject* self, PyObject* args)
{
    // Holds input and output arrays
    PyArrayObject *arrays[3];
    // The return value of the function
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[3];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[3];

    NpyIter_IterNextFunc *iternext;

    // Parse two NumPy array arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arrays[0],
                                        &PyArray_Type, &arrays[1])) {
        return NULL;
    }

    // The result will be allocated by the iterator
    arrays[2] = NULL;

    // Set up and create the iterator

    iterator_flags = (NPY_ITER_ZEROSIZE_OK |
                      // Enable buffering
                      NPY_ITER_BUFFERED |
                      // Manually handle innermost iteration for speed:
                      NPY_ITER_EXTERNAL_LOOP |
                      NPY_ITER_GROWINNER |
                      0);

    op_flags[0] = (NPY_ITER_READONLY |
                   // Required that the arrays are well behaved
                   NPY_ITER_NBO |
                   NPY_ITER_ALIGNED);
    op_flags[1] = op_flags[0];

    // Ask the iterator to allocate an array to write the output to
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    //
    // For this demonstraion, we handle only np.float64 arrays.
    // The array op_dtypes is used only as an argument to NpyIter_MultNew.
    // 
    op_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    op_dtypes[1] = op_dtypes[0];
    op_dtypes[2] = op_dtypes[0];

    // Create the NumPy iterator object:
    iter = NpyIter_MultiNew(3,  // 2 inputs, 1 output
                            arrays,
                            iterator_flags,
                            // Use input order for output and iteration
                            NPY_KEEPORDER,
                            // Allow only byte-swapping of input
                            NPY_EQUIV_CASTING,
                            op_flags,
                            op_dtypes);
    Py_DECREF(op_dtypes[0]);  // op_dtypes[1] and op_dtypes[2] are the same instance.

    if (iter == NULL)
        return NULL;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // Fetch the output array which was allocated by the iterator:
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(ret);

    if (NpyIter_GetIterSize(iter) == 0) {
        //
        // If there are no elements, the loop cannot be iterated.
        // This check is necessary with NPY_ITER_ZEROSIZE_OK.
        //
        NpyIter_Deallocate(iter);
        return ret;
    }

    // The location of the data pointer which the iterator may update
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    // The location of the stride which the iterator may update
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    // The location of the inner loop size which the iterator may update
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // Iterate over the arrays
    do {
        npy_intp count = *innersizeptr;
        npy_intp x_stride = strideptr[0];
        npy_intp y_stride = strideptr[1];
        npy_intp out_stride = strideptr[2];
#ifdef DEBUG
        printf("hypot_v1: count = %zd,  x_stride = %zd,  y_stride = %zd,  out_stride = %zd\n",
               count, x_stride, y_stride, out_stride);
#endif
        char *p_x = dataptr[0];
        char *p_y = dataptr[1];
        char *p_out = dataptr[2];

        while (count--) {
            double x = *(double *)p_x;
            double y = *(double *)p_y;
            double h = hypot(x, y);
#ifdef DEBUG
        printf("x = %23.15lf,  y = %23.15lf,  h = %23.15lf\n", x, y, h);
#endif
            *(double *)p_out = h;
            p_x += x_stride;
            p_y += y_stride;
            p_out += out_stride;
        }
    } while (iternext(iter));

    // Clean up and return the result
    NpyIter_Deallocate(iter);
    return ret;
}


static PyObject* hypot_v2(PyObject* self, PyObject* args)
{
    // Holds input and output arrays
    PyArrayObject *arrays[3];
    // The return value of the function
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[3];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[3];

    NpyIter_IterNextFunc *iternext;

    // Parse two NumPy array arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arrays[0],
                                        &PyArray_Type, &arrays[1])) {
        return NULL;
    }

    // The result will be allocated by the iterator
    arrays[2] = NULL;

    // Set up and create the iterator

    iterator_flags = NPY_ITER_ZEROSIZE_OK;

    op_flags[0] = (NPY_ITER_READONLY |
                   // Required that the arrays are well behaved
                   NPY_ITER_NBO |
                   NPY_ITER_ALIGNED);
    op_flags[1] = op_flags[0];

    // Ask the iterator to allocate an array to write the output to
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    //
    // For this demonstraion, we handle only np.float64 arrays.
    // The array op_dtypes is used only as an argument to NpyIter_MultNew.
    // 
    op_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    op_dtypes[1] = op_dtypes[0];
    op_dtypes[2] = op_dtypes[0];

    // Create the NumPy iterator object:
    iter = NpyIter_MultiNew(3,  // 2 inputs, 1 output
                            arrays,
                            iterator_flags,
                            // Use input order for output and iteration
                            NPY_KEEPORDER,
                            // Allow only byte-swapping of input
                            NPY_EQUIV_CASTING,
                            op_flags,
                            op_dtypes);
    Py_DECREF(op_dtypes[0]);  // op_dtypes[1] and op_dtypes[2] are the same instance.

    if (iter == NULL)
        return NULL;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // Fetch the output array which was allocated by the iterator:
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(ret);

    if (NpyIter_GetIterSize(iter) == 0) {
        //
        // If there are no elements, the loop cannot be iterated.
        // This check is necessary with NPY_ITER_ZEROSIZE_OK.
        //
        NpyIter_Deallocate(iter);
        return ret;
    }

    // The location of the data pointer which the iterator may update
    char **dataptr = NpyIter_GetDataPtrArray(iter);

    // Iterate over the arrays
    do {
        double x = *(double *)(dataptr[0]);
        double y = *(double *)(dataptr[1]);
        double h = hypot(x, y);
#ifdef DEBUG
        printf("x = %23.15lf,  y = %23.15lf,  h = %23.15lf\n", x, y, h);
#endif
        *(double *)(dataptr[2]) = h;
    } while (iternext(iter));

    // Clean up and return the result
    NpyIter_Deallocate(iter);
    return ret;
}

// Define functions in module
static PyMethodDef IterExamplesMethods[] =
{
     {"cos_v1", cos_v1, METH_VARARGS,
         "evaluate the cosine on a NumPy array"},
     {"cos_v2", cos_v2, METH_VARARGS,
         "evaluate the cosine on a NumPy array"},
     {"hypot_v1", hypot_v1, METH_VARARGS,
         "evaluate hypot(x, y) on NumPy arrays"},
     {"hypot_v2", hypot_v2, METH_VARARGS,
         "evaluate hypot(x, y) on NumPy arrays"},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "iter_examples",
    MODULE_DOCSTRING,
    -1,
    IterExamplesMethods
};

PyMODINIT_FUNC PyInit_iter_examples(void)
{
    PyObject *module;

    module = PyModule_Create(&module_def);
    if(module == NULL) {
        return NULL;
    }

    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }

    return module;
}
