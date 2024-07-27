#include <torch/extension.h>
#include "utils.h"

torch::Tensor rational_fwd(
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d) {
  CHECK_INPUT(x);
  CHECK_INPUT(n);
  CHECK_INPUT(d);
  return rational_fwd_cuda(x, n, d);
}

torch::Tensor rational_bwd(
  torch::Tensor dy, 
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d) {
  CHECK_INPUT(dy);
  CHECK_INPUT(x);
  CHECK_INPUT(n);
  CHECK_INPUT(d);
  return rational_bwd_cuda(dy, x, n, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rational_fwd", &rational_fwd, 
    "rational forward (CUDA)");
  m.def("rational_bwd", &rational_bwd,
    "rational backward (CUDA)");
}