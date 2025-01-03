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

Also need to rename the wheel, replace ``linux`` with ``manylinux1``.

## upload to PyPI

```bash
twine upload dist/*
```