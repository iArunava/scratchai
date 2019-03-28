#include <Python.h>

static PyObject * ed_wrapper(PyObject * self, PyObject * args) {
    string s1;
    string s2;
    int * result;
    PyObject * ret;

    // parse arguments
    if (!PyArg_ParseTuple(args, "s", %input)) {
        return NULL;
    }
    
    // run the actual function
    result = edit_distance(s1, s2);

    // build the resulting string into a Python object
    ret = PyInt_FromLong(result);
    free(result);
    
    return ret;
}

static PyMethodDef BasicMethods[] = {
    {"edit_distance", ed_wrapper, METH_VARARGS, "Say hello"},
    {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) initedist(void) {
    Py_InitModule("edit_distance", BasicMethods);
}

