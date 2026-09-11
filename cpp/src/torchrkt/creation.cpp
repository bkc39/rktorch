#include "torchrkt/c_api/creation.h"

#include <torch/torch.h>

#include <limits>
#include <utility>
#include <vector>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/dtype.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

torch::TensorOptions default_options() {
  return torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(torchrkt::current_default_device());
}

torch::TensorOptions options_on(tr_device_type type, int64_t index,
                                tr_dtype dtype) {
  torch::TensorOptions options = default_options();
  if (type != TR_DEVICE_KEEP) {
    options = options.device(torchrkt::to_torch_device(type, index));
  }
  if (dtype != TR_DTYPE_KEEP) {
    options = options.dtype(torchrkt::to_scalar_type(dtype));
  }
  return options;
}

bool bad_dims(const int64_t* dims, int64_t ndim) {
  return ndim < 0 || (ndim > 0 && !dims);
}

std::vector<int64_t> to_shape(const int64_t* dims, int64_t ndim) {
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
    return torchrkt::null_arg(who);
  }
  return torchrkt::alloc_result(
      who, [&] { return std::forward<Fn>(fn)(to_shape(dims, ndim)); });
}

template <typename T>
torch::Tensor host_from_data(const T* data, uint64_t numel, const int64_t* dims,
                             int64_t ndim, torch::ScalarType dtype) {
  const auto shape = to_shape(dims, ndim);
  uint64_t expected = 1;
  for (const int64_t d : shape) {
    if (d < 0) {
      throw std::invalid_argument("negative dimension");
    }
    const auto u = static_cast<uint64_t>(d);
    if (u != 0 && expected > std::numeric_limits<uint64_t>::max() / u) {
      throw std::invalid_argument("dimension product overflows");
    }
    expected *= u;
  }
  if (expected != numel) {
    throw std::invalid_argument("numel does not match the product of dims");
  }
  // An empty Racket vector marshals as a NULL `data`.
  if (numel == 0) {
    return torch::empty(shape, torch::TensorOptions().dtype(dtype));
  }
  // clone(): the tensor must own its storage, not borrow the caller's buffer.
  return torch::from_blob(const_cast<T*>(data), shape,
                          torch::TensorOptions().dtype(dtype))
      .clone();
}

}  // namespace

extern "C" {

tr_tensor* tr_zeros(const int64_t* dims, int64_t ndim) {
  return shaped_result("tr_zeros", dims, ndim,
                       [&](const std::vector<int64_t>& shape) {
                         return torch::zeros(shape, default_options());
                       });
}

tr_tensor* tr_ones(const int64_t* dims, int64_t ndim) {
  return shaped_result("tr_ones", dims, ndim,
                       [&](const std::vector<int64_t>& shape) {
                         return torch::ones(shape, default_options());
                       });
}

tr_tensor* tr_full(const int64_t* dims, int64_t ndim, double value) {
  return shaped_result("tr_full", dims, ndim,
                       [&](const std::vector<int64_t>& shape) {
                         return torch::full(shape, value, default_options());
                       });
}

tr_tensor* tr_arange(double start, double end, double step) {
  return torchrkt::alloc_result("tr_arange", [&] {
    return torch::arange(start, end, step, default_options());
  });
}

tr_tensor* tr_eye(int64_t n, int64_t m) {
  return torchrkt::alloc_result(
      "tr_eye", [&] { return torch::eye(n, m, default_options()); });
}

tr_tensor* tr_from_data(const float* data, uint64_t numel, const int64_t* dims,
                        int64_t ndim) {
  if ((!data && numel > 0) || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data");
  }
  return torchrkt::alloc_result("tr_from_data", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kFloat32)
        .to(torchrkt::current_default_device());
  });
}

tr_tensor* tr_from_data_i64(const int64_t* data, uint64_t numel,
                            const int64_t* dims, int64_t ndim) {
  if ((!data && numel > 0) || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_i64");
  }
  return torchrkt::alloc_result("tr_from_data_i64", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kInt64)
        .to(torchrkt::current_default_device());
  });
}

tr_tensor* tr_from_data_i64_on_device(const int64_t* data, uint64_t numel,
                                      const int64_t* dims, int64_t ndim,
                                      tr_device_type device_type,
                                      int64_t device_index) {
  if ((!data && numel > 0) || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_i64_on_device");
  }
  return torchrkt::alloc_result("tr_from_data_i64_on_device", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kInt64)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

tr_tensor* tr_zeros_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                       int64_t index, tr_dtype dtype) {
  return shaped_result(
      "tr_zeros_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::zeros(shape, options_on(type, index, dtype));
      });
}

tr_tensor* tr_ones_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                      int64_t index, tr_dtype dtype) {
  return shaped_result(
      "tr_ones_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::ones(shape, options_on(type, index, dtype));
      });
}

tr_tensor* tr_full_on(const int64_t* dims, int64_t ndim, double value,
                      tr_device_type type, int64_t index, tr_dtype dtype) {
  return shaped_result(
      "tr_full_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::full(shape, value, options_on(type, index, dtype));
      });
}

tr_tensor* tr_from_data_on_device(const float* data, uint64_t numel,
                                  const int64_t* dims, int64_t ndim,
                                  tr_device_type device_type,
                                  int64_t device_index) {
  if ((!data && numel > 0) || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_on_device");
  }
  return torchrkt::alloc_result("tr_from_data_on_device", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kFloat32)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

}  // extern "C"
