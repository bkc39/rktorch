#include "torchrkt/c_api/creation.h"

#include <torch/torch.h>

#include <limits>
#include <utility>
#include <vector>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/dtype.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/options.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

template <typename T>
torch::Tensor host_from_data(const T* data, uint64_t numel, const int64_t* dims,
                             int64_t ndim, torch::ScalarType dtype) {
  const auto shape = torchrkt::to_shape(dims, ndim);
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
  return torchrkt::shaped_result(
      "tr_zeros", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::zeros(shape, torchrkt::default_options());
      });
}

tr_tensor* tr_ones(const int64_t* dims, int64_t ndim) {
  return torchrkt::shaped_result(
      "tr_ones", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::ones(shape, torchrkt::default_options());
      });
}

tr_tensor* tr_full(const int64_t* dims, int64_t ndim, double value) {
  return torchrkt::shaped_result(
      "tr_full", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::full(shape, value, torchrkt::default_options());
      });
}

tr_tensor* tr_arange(double start, double end, double step) {
  return torchrkt::alloc_result("tr_arange", [&] {
    return torch::arange(start, end, step, torchrkt::default_options());
  });
}

tr_tensor* tr_eye(int64_t n, int64_t m) {
  return torchrkt::alloc_result(
      "tr_eye", [&] { return torch::eye(n, m, torchrkt::default_options()); });
}

tr_tensor* tr_arange_on(double start, double end, double step,
                        tr_device_type type, int64_t index, tr_dtype dtype) {
  return torchrkt::alloc_result("tr_arange_on", [&] {
    return torch::arange(start, end, step,
                         torchrkt::options_on(type, index, dtype));
  });
}

tr_tensor* tr_eye_on(int64_t n, int64_t m, tr_device_type type, int64_t index,
                     tr_dtype dtype) {
  return torchrkt::alloc_result("tr_eye_on", [&] {
    return torch::eye(n, m, torchrkt::options_on(type, index, dtype));
  });
}

tr_tensor* tr_from_data(const float* data, uint64_t numel, const int64_t* dims,
                        int64_t ndim) {
  if ((!data && numel > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data");
  }
  return torchrkt::alloc_result("tr_from_data", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kFloat32)
        .to(torchrkt::current_default_device());
  });
}

tr_tensor* tr_from_data_i64(const int64_t* data, uint64_t numel,
                            const int64_t* dims, int64_t ndim) {
  if ((!data && numel > 0) || torchrkt::bad_dims(dims, ndim)) {
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
  if ((!data && numel > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_i64_on_device");
  }
  return torchrkt::alloc_result("tr_from_data_i64_on_device", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kInt64)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

tr_tensor* tr_from_data_u8(const uint8_t* data, uint64_t numel,
                           const int64_t* dims, int64_t ndim) {
  if ((!data && numel > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_u8");
  }
  return torchrkt::alloc_result("tr_from_data_u8", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kUInt8)
        .to(torchrkt::current_default_device());
  });
}

tr_tensor* tr_from_data_u8_on_device(const uint8_t* data, uint64_t numel,
                                     const int64_t* dims, int64_t ndim,
                                     tr_device_type device_type,
                                     int64_t device_index) {
  if ((!data && numel > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_u8_on_device");
  }
  return torchrkt::alloc_result("tr_from_data_u8_on_device", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kUInt8)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

tr_tensor* tr_zeros_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                       int64_t index, tr_dtype dtype) {
  return torchrkt::shaped_result(
      "tr_zeros_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::zeros(shape, torchrkt::options_on(type, index, dtype));
      });
}

tr_tensor* tr_ones_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                      int64_t index, tr_dtype dtype) {
  return torchrkt::shaped_result(
      "tr_ones_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::ones(shape, torchrkt::options_on(type, index, dtype));
      });
}

tr_tensor* tr_full_on(const int64_t* dims, int64_t ndim, double value,
                      tr_device_type type, int64_t index, tr_dtype dtype) {
  return torchrkt::shaped_result(
      "tr_full_on", dims, ndim, [&](const std::vector<int64_t>& shape) {
        return torch::full(shape, value,
                           torchrkt::options_on(type, index, dtype));
      });
}

tr_tensor* tr_from_data_on_device(const float* data, uint64_t numel,
                                  const int64_t* dims, int64_t ndim,
                                  tr_device_type device_type,
                                  int64_t device_index) {
  if ((!data && numel > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_on_device");
  }
  return torchrkt::alloc_result("tr_from_data_on_device", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kFloat32)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

namespace {

torch::Tensor host_from_bytes(const uint8_t* data, uint64_t nbytes,
                              const int64_t* dims, int64_t ndim,
                              tr_dtype dtype) {
  const torch::ScalarType scalar = torchrkt::to_scalar_type(dtype);
  const auto size = static_cast<uint64_t>(torch::elementSize(scalar));
  if (nbytes % size != 0) {
    throw std::invalid_argument(
        "byte count is not a multiple of the element size");
  }
  // a byte other than 0 or 1 is not a bool ATen can hold, so the payload
  // is read as uint8 and compared, as numpy's astype(bool) does
  if (scalar == torch::kBool) {
    return host_from_data(data, nbytes, dims, ndim, torch::kUInt8).ne(0);
  }
  return host_from_data(data, nbytes / size, dims, ndim, scalar);
}

}  // namespace

tr_tensor* tr_from_bytes(const uint8_t* data, uint64_t nbytes,
                         const int64_t* dims, int64_t ndim, tr_dtype dtype) {
  if ((!data && nbytes > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_bytes");
  }
  return torchrkt::alloc_result("tr_from_bytes", [&] {
    return host_from_bytes(data, nbytes, dims, ndim, dtype)
        .to(torchrkt::current_default_device());
  });
}

tr_tensor* tr_from_bytes_on_device(const uint8_t* data, uint64_t nbytes,
                                   const int64_t* dims, int64_t ndim,
                                   tr_dtype dtype, tr_device_type device_type,
                                   int64_t device_index) {
  if ((!data && nbytes > 0) || torchrkt::bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_bytes_on_device");
  }
  return torchrkt::alloc_result("tr_from_bytes_on_device", [&] {
    return host_from_bytes(data, nbytes, dims, ndim, dtype)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

}  // extern "C"
