# Batch linear assignment solver for PyTorch
Batch computation of the linear assignment problem on GPU.

## Install
Build and install package:
```
pip install .
```

When building with CUDA, make sure NVCC has the same CUDA version as PyTorch.

If you need custom C++ compiler, use the following command:
```
CXX=<compiler> pip install .
```
