"""Package configuration with an opt-in legacy CUDA benchmark extension."""

import os

import setuptools


def get_build_ext_modules() -> list[object]:
    """Return the legacy CUDA extension only when a benchmark build requests it."""
    if os.environ.get("TLA_BUILD_LEGACY_CUDA") != "1":
        return []

    import torch
    import torch.utils.cpp_extension as torch_cpp_ext

    can_build_cuda = torch.backends.cuda.is_built() and (
        os.environ.get("FORCE_CUDA") == "1" or torch.cuda.is_available()
    )
    if not can_build_cuda:
        raise RuntimeError(
            "TLA_BUILD_LEGACY_CUDA=1 requires a CUDA-enabled Torch build and CUDA availability; "
            "CPU installs do not build the legacy extension."
        )

    compile_args = {"cxx": ["-O3"]}
    if os.environ.get("CC") is not None:
        compile_args["nvcc"] = ["-ccbin", os.environ["CC"]]
    return [
        torch_cpp_ext.CUDAExtension(
            "torch_linear_assignment._backend",
            [
                "src/torch_linear_assignment_cuda.cpp",
                "src/torch_linear_assignment_cuda_kernel.cu",
            ],
            extra_compile_args=compile_args,
        )
    ]


def get_build_ext() -> object:
    """Return Torch's extension command for an opted-in legacy CUDA build."""
    import torch.utils.cpp_extension as torch_cpp_ext

    return torch_cpp_ext.BuildExtension


def get_requirements() -> list[str]:
    """Read install dependencies without retaining blank requirement entries."""
    with open("requirements.txt", encoding="utf-8") as requirements_file:
        return [line.strip() for line in requirements_file if line.strip()]


def get_long_description() -> str:
    """Read the package long description from the repository README."""
    with open("README.md", encoding="utf-8") as readme_file:
        return readme_file.read()


if __name__ == "__main__":
    extension_modules = get_build_ext_modules()
    setup_kwargs: dict[str, object] = {
        "name": "torch-linear-assignment",
        "version": "0.1.0",
        "author": "Ivan Karpukhin",
        "author_email": "karpuhini@yandex.ru",
        "description": "Batched linear assignment with PyTorch and CUDA.",
        "long_description": get_long_description(),
        "long_description_content_type": "text/markdown",
        "packages": ["torch_linear_assignment"],
        "python_requires": ">=3.10",
        "ext_modules": extension_modules,
        "install_requires": get_requirements(),
    }
    if extension_modules:
        setup_kwargs["cmdclass"] = {"build_ext": get_build_ext()}
    setuptools.setup(**setup_kwargs)
