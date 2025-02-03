// file: vllm_allocator_adaptor_c.cpp
//
// An adaptor to pass Python function to PyTorch's pluggable allocator.
// Important: allocation size, CUdeviceptr and CUmemGenericAllocationHandle* need to be unsigned long long

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include <iostream>

#include <hip/hip_runtime_api.h>

extern "C" {
#ifndef CUDA_SUCCESS
    #define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS

// https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html
typedef unsigned long long CUdevice;
typedef hipDeviceptr_t CUdeviceptr;
typedef hipError_t CUresult;
typedef hipCtx_t CUcontext;
typedef hipStream_t CUstream;
typedef hipMemGenericAllocationHandle_t CUmemGenericAllocationHandle;
typedef hipMemAllocationGranularity_flags CUmemAllocationGranularity_flags;
typedef hipMemAllocationProp CUmemAllocationProp;
typedef hipMemAccessDesc CUmemAccessDesc;

#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum

// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
#define CU_MEM_ALLOCATION_COMP_NONE 0x0

// Error Handling
// https://docs.nvidia.com/cuda/archive/11.4.4/cuda-driver-api/group__CUDA__ERROR.html
CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorString(hipError);
    return CUDA_SUCCESS;
}

// Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
CUresult cuCtxGetCurrent(CUcontext *ctx) {
    // This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform.
    return hipCtxGetCurrent(ctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    // This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform.
    return hipCtxSetCurrent(ctx);
}

// Primary Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev) {
    return hipDevicePrimaryCtxRetain(ctx, dev);
}

// Virtual Memory Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    return hipMemAddressFree(ptr, size);
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    return hipMemAddressReserve(ptr, size, alignment, addr, flags);
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags ) {
    return hipMemCreate(handle, size, prop, flags);
}

CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) {
    return hipMemGetAllocationGranularity(granularity, prop, option);
}

CUresult cuMemMap(CUdeviceptr dptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return hipMemMap(dptr, size, offset, handle, flags);
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return hipMemRelease(handle);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
    return hipMemSetAccess(ptr, size, desc, count);
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    return hipMemUnmap(ptr, size);
}

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

void ensure_context(unsigned long long device)
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

void create_and_map(unsigned long long device, ssize_t size, CUdeviceptr d_mem, CUmemGenericAllocationHandle* p_memHandle)
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
    std::cout << "[vllm_allocator_adaptor_c] create_and_map: device=" << device << ", size=" << size << ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
}

void unmap_and_release(unsigned long long device, ssize_t size, CUdeviceptr d_mem, CUmemGenericAllocationHandle* p_memHandle)
{
    std::cout << "[vllm_allocator_adaptor_c] unmap_and_release: device=" << device << ", size=" << size << ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
    ensure_context(device);
    CUDA_CHECK(cuMemUnmap(d_mem, size));
    CUDA_CHECK(cuMemRelease(*p_memHandle));
}

PyObject* create_tuple_from_c_integers(unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d) {
    // Create a new tuple of size 4
    PyObject *tuple = PyTuple_New(4);
    if (!tuple) {
        return NULL; // Return NULL on failure
    }

    // Convert integers to Python objects and set them in the tuple
    PyTuple_SetItem(tuple, 0, PyLong_FromLong(a)); // Steals reference to the PyLong
    PyTuple_SetItem(tuple, 1, PyLong_FromLong(b));
    PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
    PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));

    // Note: PyTuple_SetItem "steals" a reference to each object,
    // so we do not need to Py_DECREF the PyLong objects explicitly.

    return tuple; // Return the created tuple
}

void* my_malloc(ssize_t size, int device, CUstream stream) 
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
    std::cout << "[vllm_allocator_adaptor_c] create_and_map: device=" << d_mem << ", size=" << alignedSize << std::endl;
    CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, 0, 0, 0));

    // allocate the CUmemGenericAllocationHandle
    CUmemGenericAllocationHandle* p_memHandle = (CUmemGenericAllocationHandle*)malloc(sizeof(CUmemGenericAllocationHandle));

    if (!g_python_malloc_callback) {
        std::cerr << "[vllm_allocator_adaptor_c] ERROR: g_python_malloc_callback not set.\n";
        return nullptr;
    }

    // Acquire GIL (not in stable ABI officially, but often works)
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* arg_tuple = create_tuple_from_c_integers(device, alignedSize, (unsigned long long)d_mem, (unsigned long long)p_memHandle);

    // Call g_python_malloc_callback
    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
    Py_DECREF(arg_tuple);

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

void my_free(void* ptr, ssize_t size, int device, CUstream stream)
{
    // get memory handle from the pointer
    if (!g_python_free_callback) {
        std::cerr << "[vllm_allocator_adaptor_c] ERROR: g_python_free_callback not set.\n";
        return;
    }

    // Acquire GIL (not in stable ABI officially, but often works)
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* py_ptr  = PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);

    if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
        PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
        return;
    }

    unsigned long long recv_device, recv_size;
    unsigned long long recv_d_mem, recv_p_memHandle;
    // Unpack the tuple into four C integers
    if (!PyArg_ParseTuple(py_result, "KKKK", &recv_device, &recv_size, &recv_d_mem, &recv_p_memHandle)) {
        // PyArg_ParseTuple sets an error if it fails
        return;
    }

    PyGILState_Release(gstate);

    // recv_size == size
    // recv_device == device

    // Free memory

    CUdeviceptr d_mem = (CUdeviceptr)recv_d_mem;
    CUmemGenericAllocationHandle* p_memHandle = (CUmemGenericAllocationHandle*)recv_p_memHandle;
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

static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
    if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
        return nullptr;
    }

    unsigned long long recv_device, recv_size;
    unsigned long long recv_d_mem, recv_p_memHandle;
    // Unpack the tuple into four C integers
    if (!PyArg_ParseTuple(args, "KKKK", &recv_device, &recv_size, &recv_d_mem, &recv_p_memHandle)) {
        // PyArg_ParseTuple sets an error if it fails
        return nullptr;
    }

    CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
    CUmemGenericAllocationHandle* p_memHandle = (CUmemGenericAllocationHandle*)recv_p_memHandle;

    unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);

    Py_RETURN_NONE;
}

static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
    if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
        return nullptr;
    }

    unsigned long long recv_device, recv_size;
    unsigned long long recv_d_mem, recv_p_memHandle;
    // Unpack the tuple into four C integers
    if (!PyArg_ParseTuple(args, "KKKK", &recv_device, &recv_size, &recv_d_mem, &recv_p_memHandle)) {
        // PyArg_ParseTuple sets an error if it fails
        return nullptr;
    }

    CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
    CUmemGenericAllocationHandle* p_memHandle = (CUmemGenericAllocationHandle*)recv_p_memHandle;

    create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);

    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {
        "init_module",
        (PyCFunction)py_init_module,
        METH_VARARGS,
        "Initialize module with python_malloc and python_free callables."
    },
    {
        "python_create_and_map",
        (PyCFunction)python_create_and_map,
        METH_VARARGS,
        "Create and map memory on the device."
    },
    {
        "python_unmap_and_release",
        (PyCFunction)python_unmap_and_release,
        METH_VARARGS,
        "Unmap and release memory on the device."
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
