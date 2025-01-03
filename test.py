import vllm_allocator_adaptor
import torch
from typing import Optional

def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """ # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), \
        f"Unexpected filename: {filename} for library {lib_name}"
    return path

lib_name = find_loaded_library("vllm_allocator_adaptor")

def get_pluggable_allocator(python_malloc_fn, python_free_func):
    vllm_allocator_adaptor.init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    lib_name, 'my_malloc', 'my_free')
    return new_alloc

def python_malloc(size):
    print("Python side: Alloc called with size =", size)
    # Return an integer that simulates a pointer
    return 0xDEADBEEF

def python_free(ptr, size):
    print(f"Python side: Free called with ptr=0x{ptr:x}, size={size}")

new_alloc = get_pluggable_allocator(python_malloc, python_free)

mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)

with torch.cuda.memory.use_mem_pool(mem_pool):
    x = torch.empty(2, 3, device='cuda')
    print(x)
    y = torch.empty(2, 3, device='cuda')
    print(y)
    z = torch.empty(2, 3, device='cuda')
    print(z)