import torch
import ctypes

from vllm_allocator_adaptor import use_memory_pool_with_allocator

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

cudart = CudaRTLibrary()

def python_malloc(size):
    print("Python side: Alloc called with size =", size)
    ptr = cudart.cudaMalloc(size)
    print(f"Python side: Returning ptr=0x{ptr.value:x}")
    return ptr.value

def python_free(ptr, size):
    print(f"Python side: Free called with ptr=0x{ptr:x}, size={size}")
    cudart.cudaFree(ctypes.c_void_p(ptr))

# default memory pool
x = torch.empty(2, 3, device='cuda')
print(x)

with use_memory_pool_with_allocator(python_malloc, python_free):
    # custom memory pool
    y = torch.empty(2, 3, device='cuda')
    print(y)
    z = torch.empty(2, 3, device='cuda')
    print(z)
