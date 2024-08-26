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

If you get a torch-not-found error, try the following command:
```
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
```
tensor([[ 0,  2, -1,  1]], device='cuda:0')
```

# Citation
If you use this code in your research project, please cite one of the following papers:
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
