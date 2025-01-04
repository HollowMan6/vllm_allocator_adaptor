// file: vllm_allocator_adaptor_c.cpp
//
// An adaptor to pass Python function to PyTorch's pluggable allocator.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cuda.h>

#define CUDA_CHECK(condition) \
    do { \
        CUresult error = condition; \
        if (error != 0) { \
            char* error_string; \
            cuGetErrorString(error, (const char**)&error_string); \
            std::cerr << "[vllm_allocator_adaptor_c] CUDA Error: " << error_string << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback   = nullptr;

extern "C" {

void ensure_context(int device)
{
    CUcontext pctx;
    CUDA_CHECK(cuCtxGetCurrent(&pctx));
    if (!pctx) {
        // Ensure device context.
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
        CUDA_CHECK(cuCtxSetCurrent(pctx));
    }
}

// ---------------------------------------------------------------------------
// Our exported C functions that call Python:

void create_and_map(int device, ssize_t size, CUdeviceptr d_mem, CUmemGenericAllocationHandle* p_memHandle)
{
    ensure_context(device);
    // Define memory allocation properties
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

    // Allocate memory using cuMemCreate
    CUDA_CHECK(cuMemCreate(p_memHandle, size, &prop, 0));
    CUDA_CHECK(cuMemMap(d_mem, size, 0, *p_memHandle, 0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUDA_CHECK(cuMemSetAccess(d_mem, size, &accessDesc, 1));
}

void unmap_and_release(int device, ssize_t size, CUdeviceptr d_mem, CUmemGenericAllocationHandle* p_memHandle)
{
    ensure_context(device);
    CUDA_CHECK(cuMemUnmap(d_mem, size));
    CUDA_CHECK(cuMemRelease(*p_memHandle));
}

void* my_malloc(ssize_t size, int device, cudaStream_t stream) 
{
    ensure_context(device);

    // first allocation, align the size, and reserve an address, and also allocate a CUmemGenericAllocationHandle

    // Define memory allocation properties
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

    // Check if the allocation is supported
    size_t granularity;
    CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

    CUdeviceptr d_mem;
    CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, 0, 0, 0));

    // allocate the CUmemGenericAllocationHandle
    CUmemGenericAllocationHandle* p_memHandle = (CUmemGenericAllocationHandle*)malloc(sizeof(CUmemGenericAllocationHandle));

    if (!g_python_malloc_callback) {
        std::cerr << "[vllm_allocator_adaptor_c] ERROR: g_python_malloc_callback not set.\n";
        return nullptr;
    }

    // Acquire GIL (not in stable ABI officially, but often works)
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Create Python int for 'device'
    PyObject* py_device = PyLong_FromSsize_t(device);
    if (!py_device) {
        PyGILState_Release(gstate);
        return nullptr;
    }

    // Create Python int for 'alignedSize'
    PyObject* py_alignedSize = PyLong_FromSsize_t(alignedSize);
    if (!py_alignedSize) {
        Py_DECREF(py_device);
        PyGILState_Release(gstate);
        return nullptr;
    }

    // Create Python int for 'd_mem'
    // assume size_t is the same as long long unsigned int
    PyObject* py_d_mem = PyLong_FromSsize_t(size_t(d_mem));
    if (!py_d_mem) {
        Py_DECREF(py_device);
        Py_DECREF(py_alignedSize);
        PyGILState_Release(gstate);
        return nullptr;
    }

    // Create Python int for 'p_memHandle'
    // assume size_t is the same as long long unsigned int
    PyObject* py_p_memHandle = PyLong_FromSsize_t(size_t(p_memHandle));
    if (!py_p_memHandle) {
        Py_DECREF(py_device);
        Py_DECREF(py_alignedSize);
        Py_DECREF(py_d_mem);
        PyGILState_Release(gstate);
        return nullptr;
    }

    // Call g_python_malloc_callback(py_alignedSize, py_d_mem, py_p_memHandle)
    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_malloc_callback, py_device, py_alignedSize, py_d_mem, py_p_memHandle, NULL);
    Py_DECREF(py_device);
    Py_DECREF(py_alignedSize);
    Py_DECREF(py_d_mem);
    Py_DECREF(py_p_memHandle);

    if (!py_result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return nullptr;
    }

    PyGILState_Release(gstate);

    // do the final mapping
    create_and_map(device, alignedSize, d_mem, p_memHandle);

    return (void*)d_mem;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream)
{
    // get memory handle from the pointer
    if (!g_python_free_callback) {
        std::cerr << "[vllm_allocator_adaptor_c] ERROR: g_python_free_callback not set.\n";
        return;
    }

    // Acquire GIL (not in stable ABI officially, but often works)
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* py_ptr  = PyLong_FromSize_t(reinterpret_cast<size_t>(ptr));
    PyObject* py_size = PyLong_FromSsize_t(size);

    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, py_size, NULL);

    if (!py_result) {
        PyErr_Print();
    }
    CUdeviceptr d_mem = (CUdeviceptr)ptr;
    CUmemGenericAllocationHandle* p_memHandle = (CUmemGenericAllocationHandle*)PyLong_AsSize_t(py_result);

    Py_DECREF(py_result);
    Py_DECREF(py_ptr);
    Py_DECREF(py_size);

    PyGILState_Release(gstate);

    // Free memory
    unmap_and_release(device, size, d_mem, p_memHandle);

    // free address and the handle
    CUDA_CHECK(cuMemAddressFree(d_mem, size));
    free(p_memHandle);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args)
{
    PyObject* malloc_callback = nullptr;
    PyObject* free_callback   = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
        return nullptr;
    }

    if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
        return nullptr;
    }

    // Save the Python callables
    // This module does not handle GC of these objects, so they must be kept alive
    // outside of this module.
    g_python_malloc_callback = malloc_callback;
    g_python_free_callback   = free_callback;

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
