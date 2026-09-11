#include "torchrkt/c_api/random.h"

#include <torch/torch.h>

#include <exception>
#include <string>
#include <vector>

#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/options.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

extern "C" {

tr_tensor* tr_randn(const int64_t* dims, int64_t ndim) {
  if (ndim < 0 || (ndim > 0 && !dims)) {
    torchrkt::set_error("tr_randn: ndim/dims inconsistent");
    return nullptr;
  }
  return torchrkt::alloc_result("tr_randn", [&] {
    const std::vector<int64_t> shape(dims, dims + ndim);
    return torch::randn(shape, torchrkt::default_options());
  });
}

tr_tensor* tr_rand(const int64_t* dims, int64_t ndim) {
  if (ndim < 0 || (ndim > 0 && !dims)) {
    torchrkt::set_error("tr_rand: ndim/dims inconsistent");
    return nullptr;
  }
  return torchrkt::alloc_result("tr_rand", [&] {
    const std::vector<int64_t> shape(dims, dims + ndim);
    return torch::rand(shape, torchrkt::default_options());
  });
}

tr_tensor* tr_randn_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                       int64_t index, tr_dtype dtype) {
  return torchrkt::shaped_result(
      "tr_randn_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::randn(shape, torchrkt::options_on(type, index, dtype));
      });
}

tr_tensor* tr_rand_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                      int64_t index, tr_dtype dtype) {
  return torchrkt::shaped_result(
      "tr_rand_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::rand(shape, torchrkt::options_on(type, index, dtype));
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
