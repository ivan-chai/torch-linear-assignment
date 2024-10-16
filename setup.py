import os
import setuptools


def get_build_ext_modules():
    import torch
    import torch.utils.cpp_extension as torch_cpp_ext

    if torch.backends.cuda.is_built() and int(os.environ.get("TLA_BUILD_CUDA", "1")) and torch.cuda.is_available():
        compile_args = {"cxx": ["-O3"]}
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
    return [
        torch_cpp_ext.CppExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"]}
        )
    ]


with open("requirements.txt", "r") as fp:
    required_packages = [line.strip() for line in fp.readlines()]


def get_build_ext():
    import torch.utils.cpp_extension as torch_cpp_ext

    return torch_cpp_ext.BuildExtension


if __name__ == '__main__':
    setuptools.setup(
        name="torch-linear-assignment",
        version="0.0.1.post2",
        author="Ivan Karpukhin",
        author_email="karpuhini@yandex.ru",
        description="Batched linear assignment with PyTorch and CUDA.",
        packages=["torch_linear_assignment"],
        ext_modules=get_build_ext_modules(),
        install_requires=required_packages,
        cmdclass={
            "build_ext": get_build_ext()
        }
    )
