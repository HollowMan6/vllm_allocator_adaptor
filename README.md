# vllm_allocator_adaptor
An adaptor to allow Python allocator for PyTorch pluggable allocator

## create source distribution

```bash
python setup.py sdist
```

## create wheel distribution

```bash
python setup.py bdist_wheel --py-limited-api cp38
```