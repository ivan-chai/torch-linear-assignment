# Batch linear assignment for PyTorch
[![PyPI version](https://badge.fury.io/py/torch-linear-assignment.svg)](https://badge.fury.io/py/torch-linear-assignment)
[![Build Status](https://github.com/ivan-chai/torch-linear-assignment/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ivan-chai/torch-linear-assignment/actions)
[![Downloads](https://static.pepy.tech/badge/torch-linear-assignment)](https://pepy.tech/project/torch-linear-assignment)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<h4 align="left">
    <p>
        <a href="#Install">Installation</a> |
        <a href="#Example">Usage</a> |
        <a href="#Citation">Citation</a>
    <p>
</h4>
Batch computation of the linear assignment problem on GPU.

## Install
Build and install via PyPI (source distribution):
```bash
pip install torch-linear-assignment
```

Build and install from Git repository:
```bash
pip install .
```

When building with CUDA, make sure NVCC has the same CUDA version as PyTorch.
You can choose CUDA version by
```bash
export PATH=/usr/local/cuda-<version>/bin:"$PATH"
```

If you need custom C++ compiler, use the following command:
```bash
CXX=<c++-compiler> CC=<gcc-compiler> pip install .
```

If you get a torch-not-found error, try the following command:
```bash
pip install --upgrade pip wheel setuptools
python -m pip install .
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
```py
tensor([[ 0,  2, -1,  1]], device='cuda:0')
```

To get indices in the SciPy's format:
```py
from torch_linear_assignment import assignment_to_indices

row_ind, col_ind = assignment_to_indices(assignment)
print(row_ind)
print(col_ind)
```

The output is:
```py
tensor([[0, 1, 3]], device='cuda:0')
tensor([[0, 2, 1]], device='cuda:0')
```

# Citation
The code was originally developed for the [HoTPP Benchmark](https://github.com/ivan-chai/hotpp-benchmark). If you use this code in your research project, please cite one of the following papers:
```
@article{karpukhin2024hotppbenchmark,
  title={HoTPP Benchmark: Are We Good at the Long Horizon Events Forecasting?},
  author={Karpukhin, Ivan and Shipilov, Foma and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2406.14341},
  year={2024},
  url ={https://arxiv.org/abs/2406.14341}
}

@article{karpukhin2024detpp,
  title={DeTPP: Leveraging Object Detection for Robust Long-Horizon Event Prediction},
  author={Karpukhin, Ivan and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2408.13131},
  year={2024},
  url ={https://arxiv.org/abs/2408.13131}
}
```
