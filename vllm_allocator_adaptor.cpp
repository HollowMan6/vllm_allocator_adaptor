// file: vllm_allocator_adaptor_c.cpp
//
// An adaptor to pass Python function to PyTorch's pluggable allocator.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
static PyObject* g_python_malloc = nullptr;
static PyObject* g_python_free   = nullptr;

extern "C" {

// ---------------------------------------------------------------------------
// Our exported C functions that call Python:

void* my_malloc(ssize_t size, int device, cudaStream_t stream) 
{

    // device and stream are not used.
    // we will just use the current device and stream.

    if (!g_python_malloc) {
        std::cerr << "[vllm_allocator_adaptor_c] ERROR: g_python_malloc not set.\n";
        return nullptr;
    }

    // Acquire GIL (not in stable ABI officially, but often works)
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Create Python int for 'size'
    PyObject* py_size = PyLong_FromSsize_t(size);
    if (!py_size) {
        PyGILState_Release(gstate);
        return nullptr;
    }

    // Call python_malloc(py_size)
    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_malloc, py_size, NULL);
    Py_DECREF(py_size);

    if (!py_result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return nullptr;
    }

    // Expect an integer pointer
    void* ptr = reinterpret_cast<void*>(PyLong_AsSize_t(py_result));
    Py_DECREF(py_result);

    PyGILState_Release(gstate);
    return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream)
{
    // device and stream are not used.
    // we will just use the current device and stream.

    if (!g_python_free) {
        std::cerr << "[vllm_allocator_adaptor_c] ERROR: g_python_free not set.\n";
        return;
    }

    // Acquire GIL (not in stable ABI officially, but often works)
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* py_ptr  = PyLong_FromSize_t(reinterpret_cast<size_t>(ptr));
    PyObject* py_size = PyLong_FromSsize_t(size);

    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_free, py_ptr, py_size, NULL);

    if (!py_result) {
        PyErr_Print();
    } else {
        Py_DECREF(py_result);
    }

    Py_DECREF(py_ptr);
    Py_DECREF(py_size);

    PyGILState_Release(gstate);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args)
{
    PyObject* malloc_func = nullptr;
    PyObject* free_func   = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &malloc_func, &free_func)) {
        return nullptr;
    }

    if (!PyCallable_Check(malloc_func) || !PyCallable_Check(free_func)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
        return nullptr;
    }

    // Save the Python callables
    // This module does not handle GC of these objects, so they must be kept alive
    // outside of this module.
    g_python_malloc = malloc_func;
    g_python_free   = free_func;

    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {
        "init_module",
        (PyCFunction)py_init_module,
        METH_VARARGS,
        "Initialize module with python_malloc and python_free callables."
    },
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef vllm_allocator_adaptor_c_module = {
    PyModuleDef_HEAD_INIT,
    "vllm_allocator_adaptor_c",
    "vLLM Allocator Adaptor",
    -1,
    module_methods
};

PyMODINIT_FUNC
PyInit_vllm_allocator_adaptor_c(void)
{
    // Initialize the module
    PyObject* module = PyModule_Create(&vllm_allocator_adaptor_c_module);
    if (!module) {
        return NULL;
    }
    return module;
}
