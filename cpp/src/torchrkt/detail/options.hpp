#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"
#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/dtype.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace torchrkt {

inline torch::TensorOptions default_options() {
  return torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(current_default_device());
}

inline torch::TensorOptions options_on(tr_device_type type, int64_t index,
                                       tr_dtype dtype) {
  torch::TensorOptions options = default_options();
  if (type != TR_DEVICE_KEEP) {
    options = options.device(to_torch_device(type, index));
  }
  if (dtype != TR_DTYPE_KEEP) {
    options = options.dtype(to_scalar_type(dtype));
  }
  return options;
}

inline bool bad_dims(const int64_t* dims, int64_t ndim) {
  return ndim < 0 || (ndim > 0 && !dims);
}

inline std::vector<int64_t> to_shape(const int64_t* dims, int64_t ndim) {
  if (ndim == 0) {
    return {};
  }
  return {dims, dims + ndim};
}

// the shared shape-constructor boundary: dims validation, then the
// allocation under the error guard; fn receives the validated shape
template <typename Fn>
tr_tensor* shaped_result(const char* who, const int64_t* dims, int64_t ndim,
                         Fn&& fn) {
  if (bad_dims(dims, ndim)) {
    return null_arg(who);
  }
  return alloc_result(
      who, [&] { return std::forward<Fn>(fn)(to_shape(dims, ndim)); });
}

}  // namespace torchrkt
