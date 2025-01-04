import torch
import ctypes

from typing import Tuple, Dict
from vllm_allocator_adaptor import use_memory_pool_with_allocator, HandleType, create_and_map, unmap_and_release

import torch
from contextlib import contextmanager

# # allocate ptr
# result, ptr = driver.cuMemAddressReserve(size, 0, 0, 0)
# assert result.value == 0

# # allocate the handle
# prop = driver.CUmemAllocationProp()
# prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
# prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
# prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
# prop.location.id = torch.cuda.current_device()
# prop.win32HandleMetaData = 0 # this is very critical, cannot use ctypes.c_void_p(0)
# result, handle = driver.cuMemCreate(size, prop, 0)
# assert result.value == 0

# # map the memory
# result, = driver.cuMemMap(ptr, size, 0, handle, 0)
# assert result.value == 0

# # set access
# access = driver.CUmemAccessDesc()
# access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
# access.location.id = torch.cuda.current_device()
# access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
# result, = driver.cuMemSetAccess(ptr, size, [access], 1)
# assert result.value == 0

# print(f"Python side: Malloc called with size={size}, ptr=0x{ptr:x}")
# return ptr.value


class CuMemAllocator:
    def __init__(self):
        self.pointer_to_handle: Dict[int, HandleType] = {}

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        py_d_mem = allocation_handle[2]
        self.pointer_to_handle[py_d_mem] = allocation_handle
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        return self.pointer_to_handle.pop(ptr)

    def unmap(self):
        for handle in self.pointer_to_handle.values():
            unmap_and_release(handle)

    def remap(self):
        for handle in self.pointer_to_handle.values():
            create_and_map(handle)

    @contextmanager
    def use_memory_pool(self):
        with use_memory_pool_with_allocator(self.python_malloc_callback, self.python_free_callback):
            yield

# default memory pool
shape = (1024, 1024)
x = torch.empty(shape, device='cuda')
x.zero_()
print(x)

allocator = CuMemAllocator()
with allocator.use_memory_pool():
    # custom memory pool
    y = torch.empty(shape, device='cuda')
    y.zero_()
    y += 1
    print(y)
    z = torch.empty(shape, device='cuda')
    z.zero_()
    z += 2
    print(z)

allocator.unmap()
allocator.remap()

output = x + y + z
print(output)
