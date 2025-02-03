# file: setup.py
from setuptools import setup, Extension

# Adjust this if needed for your CUDA install:
cuda_include_dir = "/appl/lumi/SW/CrayEnv/EB/rocm/6.2.2/include/"

module = Extension(
    name="vllm_allocator_adaptor_c",
    sources=["vllm_allocator_adaptor_c.cpp"],
    include_dirs=[cuda_include_dir],  # Only your CUDA path, Python is auto-detected
    extra_compile_args=["-fPIC", "-D", "__HIP_PLATFORM_AMD__"],
    libraries=["amdhip64"],  # Link against the libcuda library
    # Tell setuptools we want an abi3 wheel (for CPython >=3.8):
    py_limited_api=True,
    define_macros=[("Py_LIMITED_API", "0x03080000")],
)

setup(
    name="vllm_allocator_adaptor",
    version="0.4.3",
    description="vLLM Allocator Adaptor (C/CUDA/Python) using callback shims",
    python_requires=">=3.8",
    ext_modules=[module],
    packages=["vllm_allocator_adaptor"],
)