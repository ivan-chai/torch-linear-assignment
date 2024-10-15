import os
from setuptools import setup


with open("requirements.txt", "r") as fp:
    required_packages = [line.strip() for line in fp.readlines()]


def is_cuda() -> bool:
    import torch

    return torch.backends.cuda.is_built() and int(os.environ.get("TLA_BUILD_CUDA", "1")) and torch.cuda.is_available()


def generate_cuda_ext_modules() -> list:
    import torch.utils.cpp_extension as torch_cpp_ext

    compile_args = {
        "cxx": ["-O3"]
    }
    if os.environ.get("CC", None) is not None:
        compile_args["nvcc"] = ["-ccbin", os.environ["CC"]]
    return [
        torch_cpp_ext.CUDAExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment_cuda.cpp",
                "src/torch_linear_assignment_cuda_kernel.cu"
            ],
            extra_compile_args=compile_args
        )
    ]


def generate_cpu_ext_modules() -> list:
    import torch.utils.cpp_extension as torch_cpp_ext

    return [
        torch_cpp_ext.CppExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"]}
        )
    ]

def get_build_ext():
    import torch.utils.cpp_extension as torch_cpp_ext

    return torch_cpp_ext.BuildExtension


if __name__ == '__main__':
    setup(
        name="torch-linear-assignment",
        version="0.0.1.post1",
        author="Ivan Karpukhin",
        author_email="karpuhini@yandex.ru",
        description="Batched linear assignment with PyTorch and CUDA.",
        packages=["torch_linear_assignment"],
        ext_modules=generate_cuda_ext_modules() if is_cuda() else generate_cpu_ext_modules(),
        setup_requires=required_packages,
        install_requires=required_packages,
        cmdclass={
            "build_ext": get_build_ext()
        }
    )
