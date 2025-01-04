# file: setup.py
from setuptools import setup, Extension
import sys
from torch.utils.cpp_extension import CUDA_HOME

# Adjust this if needed for your CUDA install:
if CUDA_HOME is not None:
    cuda_include_dir = CUDA_HOME + "/include"
else:
    cuda_include_dir = "/usr/local/cuda/include"

module = Extension(
    name="vllm_allocator_adaptor_c",
    sources=["vllm_allocator_adaptor_c.cpp"],
    include_dirs=[cuda_include_dir],  # Only your CUDA path, Python is auto-detected
    extra_compile_args=["-fPIC"],
    libraries=["cuda"],  # Link against the libcuda library
    # Tell setuptools we want an abi3 wheel (for CPython >=3.8):
    py_limited_api=True,
    define_macros=[("Py_LIMITED_API", "0x03080000")],
)

setup(
    name="vllm_allocator_adaptor",
    version="0.2.1",
    description="vLLM Allocator Adaptor (C/CUDA/Python) using callback shims",
    python_requires=">=3.8",
    ext_modules=[module],
    packages=["vllm_allocator_adaptor"],
)