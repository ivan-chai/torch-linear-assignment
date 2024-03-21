#include <torch/extension.h>

bool has_cuda() {
  return false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("has_cuda", &has_cuda, "CUDA build flag.");
}
