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
CXX=<c++-compiler> CC=<gcc-compiler> pip install .
```

## Example
```python
import torch
from torch_linear_assignment import batch_linear_assignment

cost = torch.tensor([
    8, 4, 7,
    5, 2, 3,
    9, 6, 7,
    9, 4, 8
]).reshape(1, 4, 3).cuda()

assignment = batch_linear_assignment(cost)
print(assignment)
```

The output is:
```
tensor([[ 0,  2, -1,  1]], device='cuda:0')
```
