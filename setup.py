import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


BUILD_CUDA = torch.backends.cuda.is_built() and int(os.environ.get("TLA_BUILD_CUDA", "1"))


if BUILD_CUDA:
    compile_args = {
        "cxx": ["-O3"]
    }
    if os.environ.get("CC", None) is not None:
        compile_args["nvcc"] = ["-ccbin", os.environ["CC"]]
    ext_modules = [
        CUDAExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment_cuda.cpp",
                "src/torch_linear_assignment_cuda_kernel.cu"
            ],
            extra_compile_args=compile_args
        )
    ]
else:
    ext_modules = [
        CppExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"]}
        )
    ]


setup(
    name="torch-linear-assignment",
    version="0.0.1",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    description="Batched linear assignment with PyTorch and CUDA.",
    packages=["torch_linear_assignment"],
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension
    }
)
