# cumem-based pytorch pluggable allocator
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch. they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import torch
import ctypes

from typing import Tuple, Dict, Optional
from vllm_allocator_adaptor import use_memory_pool_with_allocator, HandleType, create_and_map, unmap_and_release

import torch
from contextlib import contextmanager

class CuMemAllocator:
    def __init__(self):
        self.pointer_to_handle: Dict[int, HandleType] = {}
        self.pointer_to_cpu_pointer: Dict[int, Optional[int]] = {}

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        py_d_mem = allocation_handle[2]
        self.pointer_to_handle[py_d_mem] = allocation_handle
        self.pointer_to_cpu_pointer[py_d_mem] = None
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        cpu_ptr = self.pointer_to_cpu_pointer.pop(ptr)
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
