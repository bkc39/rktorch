#include "torchrkt/c_api/random.h"

#include <torch/torch.h>

#include <exception>
#include <string>
#include <vector>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

torch::TensorOptions default_options() {
  return torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(torchrkt::current_default_device());
}

}  // namespace

extern "C" {

tr_tensor* tr_randn(const int64_t* dims, int64_t ndim) {
  if (ndim < 0 || (ndim > 0 && !dims)) {
    torchrkt::set_error("tr_randn: ndim/dims inconsistent");
    return nullptr;
  }
  return torchrkt::alloc_result("tr_randn", [&] {
    const std::vector<int64_t> shape(dims, dims + ndim);
    return torch::randn(shape, default_options());
  });
}

tr_tensor* tr_rand(const int64_t* dims, int64_t ndim) {
  if (ndim < 0 || (ndim > 0 && !dims)) {
    torchrkt::set_error("tr_rand: ndim/dims inconsistent");
    return nullptr;
  }
  return torchrkt::alloc_result("tr_rand", [&] {
    const std::vector<int64_t> shape(dims, dims + ndim);
    return torch::rand(shape, default_options());
  });
}

int tr_tensor_uniform_(tr_tensor* t, double low, double high) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_uniform_");
  }
  return torchrkt::status_call("tr_tensor_uniform_",
                               [&] { t->value.uniform_(low, high); });
}

}  // extern "C"
