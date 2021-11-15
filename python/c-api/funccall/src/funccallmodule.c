
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>
#include <assert.h>

#include <stdio.h>

//
// XXX Check use of strcmp with UTF-8 strings (only testing for equality).
//
static Py_ssize_t
find_name_in_list(char **list, Py_ssize_t nlist, const char *name)
{
    Py_ssize_t k = 0;

    while (k < nlist && strcmp(list[k], name) != 0) {
        ++k;
    }
    k = (k == nlist) ? -1 : k;
    return k;
}

//
// process_args
//
// The first three arguments are the arguments that were passed by Python
// to the function with the METH_FASTCALL | METH_KEYWORDS flags.
//
// The argument processing corresponds to a function signature such as
//
//     def func(a, b, c, *, d, e, f=None, g=0)
//
// a, b, and c are the positional arguments (but they may also be given
// with keyword assignment); d and e are not optional, and must be given
// with keyword assignment; f and g are optional (and if given, must also
// be given with keyword assignment).
//
// The function does not allow for a signature that corresponds to
// using `/` in the `def` statement (i.e. it does not allow positional *only*
// arguments, with no keyword assignment allowed).
//
// `allargs` is the output array.  It must be an array of PyObject
// pointers with length
// `n_pos_arg_names + n_kw_arg_names + n_optional_kw_arg_names`.
// If there is no error, the pointers will be set to the objects passed
// in, or to a given default value.  The refcounts of all objects assigned
// to allargs are incremented.
//
// Returns 0 on success, or a negative value on failure.
//
static int
process_args(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
             const char *funcname,
             char **pos_arg_names, Py_ssize_t n_pos_arg_names,
             char **kw_arg_names, Py_ssize_t n_kw_arg_names,
             char **optional_kw_arg_names, Py_ssize_t n_optional_kw_arg_names,
             PyObject **default_kw_arg_values,
             PyObject **allargs)
{
    Py_ssize_t n_allargs = n_pos_arg_names + n_kw_arg_names + n_optional_kw_arg_names;
    Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;

    if (nargs > n_pos_arg_names) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most %ld positional "
                                      "arguments, but %ld were given",
                     funcname, n_pos_arg_names, nargs);
        return -1;
    }
    for (Py_ssize_t i = 0; i < n_allargs; ++i) {
        allargs[i] = NULL;
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        allargs[i] = args[i];
    }

    for (Py_ssize_t k = 0; k < nkw; ++k) {
        // k is the index into kwnames.  i is the index into args.
        Py_ssize_t i = nargs + k;
        // Three cases for parameters given with keywords:
        // 1. It is a parameter that can also be positional.
        // 2. It is required parameter with no default.
        // 3. It is an optional parameter.
        Py_ssize_t pos;
        PyObject *item = PyTuple_GET_ITEM(kwnames, k);
        // item must be a str (or subclass).
        const char *name = PyUnicode_AsUTF8(item);
        pos = find_name_in_list(pos_arg_names, n_pos_arg_names, name);
        if (pos >= 0) {
            // The name is one of the positional arguments.
            if (allargs[pos] != NULL) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'",
                             funcname, name);
                return -2;
            }
            allargs[pos] = args[i];
        }
        else {
            pos = find_name_in_list(kw_arg_names, n_kw_arg_names, name);
            if (pos >= 0) {
                // The name is a required keyword parameter.
                allargs[n_pos_arg_names + pos] = args[i];
            }
            else {
                pos = find_name_in_list(optional_kw_arg_names, n_optional_kw_arg_names, name);
                if (pos >= 0) {
                    // The name is an optional keyword parameter.
                    allargs[n_pos_arg_names + n_kw_arg_names + pos] = args[i];
                }
                else {
                    // The name is not one of the known names.
                    PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%s'",
                                 funcname, name);
                    return -3;
                }
            }
        }
    }

    for (Py_ssize_t i = 0; i < n_pos_arg_names + n_kw_arg_names; ++i) {
        if (allargs[i] == NULL) {
            char *name;
            if (i < n_pos_arg_names) {
                name = pos_arg_names[i];
            }
            else {
                name = kw_arg_names[i - n_pos_arg_names];
            }
            PyErr_Format(PyExc_TypeError, "%s() missing required argument '%s'",
                         funcname, name);
            return -4;
        }
    }

    // Fill in the default values of any optional keyword parameters
    // that were not given.
    for (Py_ssize_t k = 0; k < n_optional_kw_arg_names; ++k) {
        Py_ssize_t i = k + n_pos_arg_names + n_kw_arg_names;
        if (allargs[i] == NULL) {
            allargs[i] = default_kw_arg_values[k];
        }
    }

    for (Py_ssize_t i = 0; i < n_allargs; ++i) {
        assert(allargs[i] != NULL);
        Py_INCREF(allargs[i]);
    }

    return 0;
}

static PyObject *
func1(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *items = NULL;
    PyObject *n = NULL;
    PyObject *p = Py_None;
    static char *kwlist[] = {"items", "n", "p", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|$O:func1", kwlist,
                                     &items, &n, &p)) {
        return NULL;
    }

    printf("items:\n");
    PyObject_Print(items, stdout, 0);
    printf("\n");
    printf("n:\n");
    PyObject_Print(n, stdout, 0);
    printf("\n");
    printf("p:\n");
    PyObject_Print(p, stdout, 0);
    printf("\n");

    Py_RETURN_NONE;
}

#define FUNC1_DOCSTRING \
"func1(items, n, *, p=None)\n"\
"\n"\
"This is the docstring for func1().\n"\
""

static PyObject *
func2(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *allargs[3];
    char *pos_arg_names[2] = {"items", "n"};
    char *kw_arg_names[0] = {};
    char *optional_kw_arg_names[1] = {"p"};
    PyObject *default_kw_values[1] = {Py_None};
    Py_INCREF(Py_None);

    int status = process_args(args, nargs, kwnames,
                              "func2",
                              pos_arg_names, 2,
                              kw_arg_names, 0,
                              optional_kw_arg_names, 1,
                              default_kw_values,
                              allargs);

    if (status < 0) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < 3; ++i) {
        printf("allargs[%ld] = ", i);
        PyObject_Print(allargs[i], stdout, 0);
        printf("\n");
    }

    for (Py_ssize_t i = 0; i < 3; ++i) {
        Py_DECREF(allargs[i]);
    }

    Py_RETURN_NONE;
}

#define FUNC2_DOCSTRING \
"func2(items, n, *, p=None)\n"\
"\n"\
"This is the docstring for func2().\n"\
""


static PyMethodDef funccall_methods[] = {
    {"func1", (PyCFunction)(void(*)(void)) func1,
     METH_VARARGS | METH_KEYWORDS,
     FUNC1_DOCSTRING},
    {"func2", (_PyCFunctionFastWithKeywords)(void(*)(void)) func2,
     METH_FASTCALL | METH_KEYWORDS,
     FUNC2_DOCSTRING},
    {NULL, NULL, 0, NULL}
};

#define FUNCCALL_MODULE_DOCSTRING \
"Docstring for the funccall module.\n"\
"\n"\
"The module defines some functions.\n"\
""

static struct PyModuleDef funccallmodule = {
    PyModuleDef_HEAD_INIT,
    "funccall",
    FUNCCALL_MODULE_DOCSTRING,
    -1,
    funccall_methods
};

PyMODINIT_FUNC
PyInit_funccall(void)
{
    PyObject *module;

    module = PyModule_Create(&funccallmodule);
    if (module == NULL) {
        return NULL;
    }

    return module;
}
