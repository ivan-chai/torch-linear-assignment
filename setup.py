import os
from setuptools import setup

import torch
import torch.utils.cpp_extension as torch_cpp_ext


BUILD_CUDA = torch.backends.cuda.is_built() and int(os.environ.get("TLA_BUILD_CUDA", "1"))

with open("requirements.txt", "r") as fp:
    required_packages = [line.strip() for line in fp.readlines()]

if BUILD_CUDA:
    compile_args = {
        "cxx": ["-O3"]
    }
    if os.environ.get("CC", None) is not None:
        compile_args["nvcc"] = ["-ccbin", os.environ["CC"]]
    ext_modules = [
        torch_cpp_ext.CUDAExtension(
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
        torch_cpp_ext.CppExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"]}
        )
    ]


if __name__ == '__main__':
    setup(
        name="torch-linear-assignment",
        version="0.0.1",
        author="Ivan Karpukhin",
        author_email="karpuhini@yandex.ru",
        description="Batched linear assignment with PyTorch and CUDA.",
        packages=["torch_linear_assignment"],
        ext_modules=ext_modules,
        install_requires=required_packages,
        cmdclass={
            "build_ext": torch_cpp_ext.BuildExtension
        }
    )
