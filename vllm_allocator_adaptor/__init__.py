import vllm_allocator_adaptor_c
import torch

from typing import Optional, Callable
from contextlib import contextmanager

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

lib_name = find_loaded_library("vllm_allocator_adaptor_c")

if lib_name is None:
    raise RuntimeError("vLLM Allocator Adaptor library not found in the process memory map")

def get_pluggable_allocator(python_malloc_fn: Callable[[int], int], python_free_func: Callable[[int, int], None]) -> torch.cuda.memory.CUDAPluggableAllocator:
    vllm_allocator_adaptor_c.init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    lib_name, 'my_malloc', 'my_free')
    return new_alloc

@contextmanager
def use_memory_pool_with_allocator(python_malloc_fn: Callable[[int], int], python_free_func: Callable[[int, int], None]) -> None:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool
