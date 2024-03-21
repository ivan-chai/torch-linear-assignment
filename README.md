# Batch linear assignment for PyTorch
Batch computation of the linear assignment problem on GPU.

## Install
Build and install package:
```
pip install .
```

When building with CUDA, make sure NVCC has the same CUDA version as PyTorch.
You can choose CUDA version by
```
export PATH=/usr/local/cuda-<version>/bin:"$PATH"
```

If you need custom C++ compiler, use the following command:
```
CXX=<compiler> pip install .
```
