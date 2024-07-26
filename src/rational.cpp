#include <torch/extension.h>
#include 'utils.h'

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  return x + y;
}

torch::Tensor add_cpu(torch::Tensor x, torch::Tensor y) {
  return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_cpu", &add_cpu, "add two tensors");
  m.def("add_cuda", &add_cuda, "add two tensors");
}