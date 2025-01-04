import torch
import ctypes

from typing import Tuple
from vllm_allocator_adaptor import use_memory_pool_with_allocator

from cuda.bindings import driver
import torch

pointer_to_data = {}

HandleType = Tuple[int, int, int, int]

def python_malloc_callback(allocation_handle: HandleType) -> None:
    py_device, py_alignedSize, py_d_mem, py_p_memHandle = allocation_handle
    print(f"{(py_device, py_alignedSize, py_d_mem, py_p_memHandle)=}")
    global pointer_to_data
    pointer_to_data[py_d_mem] = (py_device, py_alignedSize, py_d_mem, py_p_memHandle)
    return
    # allocate ptr
    result, ptr = driver.cuMemAddressReserve(size, 0, 0, 0)
    assert result.value == 0

    # allocate the handle
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    prop.win32HandleMetaData = 0 # this is very critical, cannot use ctypes.c_void_p(0)
    result, handle = driver.cuMemCreate(size, prop, 0)
    assert result.value == 0

    # map the memory
    result, = driver.cuMemMap(ptr, size, 0, handle, 0)
    assert result.value == 0

    # set access
    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = torch.cuda.current_device()
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    result, = driver.cuMemSetAccess(ptr, size, [access], 1)
    assert result.value == 0

    print(f"Python side: Malloc called with size={size}, ptr=0x{ptr:x}")
    return ptr.value

def python_free_callback(ptr: int) -> HandleType:
    global pointer_to_data
    return pointer_to_data.pop(ptr)

# default memory pool
shape = (1024, 1024)
x = torch.empty(shape, device='cuda')
x.zero_()
print(x)

with use_memory_pool_with_allocator(python_malloc_callback, python_free_callback):
    # custom memory pool
    y = torch.empty(shape, device='cuda')
    y.zero_()
    y += 1
    print(y)
    z = torch.empty(shape, device='cuda')
    z.zero_()
    z += 2
    print(z)

output = x + y + z
print(output)
