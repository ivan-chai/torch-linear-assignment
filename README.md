# Batch linear assignment for PyTorch
<h4 align="left">
    <p>
        <a href="#Install">Installation</a> |
        <a href="#Example">Usage</a> |
        <a href="#Citation">Citation</a>
    <p>
</h4>
Batch computation of the linear assignment problem on GPU.

## Install
Build and install via PyPI:
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
