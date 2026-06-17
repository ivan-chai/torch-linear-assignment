import os
import sys
import setuptools


def get_build_ext_modules():
    import torch
    import torch.utils.cpp_extension as torch_cpp_ext

    if torch.backends.cuda.is_built() and int(os.environ.get("TLA_BUILD_CUDA", "1")) and (os.environ.get("FORCE_CUDA") == "1" or torch.cuda.is_available()):
        compile_args = {"cxx": ["-O3"]}
        if os.environ.get("CC", None) is not None:
            compile_args["nvcc"] = ["-ccbin", os.environ["CC"]]
        extra_link_args = []
        if sys.platform == "win32":
            # c10.dll does not export the inherited c10::ValueError(SourceLocation,
            # string) constructor (MSVC does not re-export inherited constructors
            # even for C10_API classes).  Headers pulled in via <torch/extension.h>
            # trigger TORCH_CHECK_VALUE which generates a __declspec(dllimport)
            # reference to that constructor, causing LNK2001.  Redirect the missing
            # dllimport thunk to Error(SourceLocation, string) which IS exported.
            # ValueError IS-A Error with no extra data members.
            # Fixed upstream by pytorch/pytorch#175340 (landed 2026-06-02), which
            # exports the ValueError/NotImplementedError constructors on Windows;
            # this workaround can be dropped once that fix is in the minimum
            # supported PyTorch.
            _val_imp = (
                "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
                "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
            )
            _err_imp = (
                "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
                "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
            )
            extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")
        return [
            torch_cpp_ext.CUDAExtension(
                "torch_linear_assignment._backend",
                [
                    "src/torch_linear_assignment_cuda.cpp",
                    "src/torch_linear_assignment_cuda_kernel.cu"
                ],
                extra_compile_args=compile_args,
                extra_link_args=extra_link_args,
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


with open("README.md") as fp:
    long_description = fp.read()


if __name__ == '__main__':
    setuptools.setup(
        name="torch-linear-assignment",
        version="0.0.6",
        author="Ivan Karpukhin",
        author_email="karpuhini@yandex.ru",
        description="Batched linear assignment with PyTorch and CUDA.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=["torch_linear_assignment"],
        ext_modules=get_build_ext_modules(),
        install_requires=required_packages,
        cmdclass={
            "build_ext": get_build_ext()
        }
    )
