#include "torchrkt/c_api/creation.h"

#include <torch/torch.h>

#include <limits>
#include <vector>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

// float32 on the process default device (CPU until tr_set_default_device flips
// it to CUDA); tr_randn shares this convention via current_default_device.
torch::TensorOptions default_options() {
  return torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(torchrkt::current_default_device());
}

bool bad_dims(const int64_t* dims, int64_t ndim) {
  return ndim < 0 || (ndim > 0 && !dims);
}

std::vector<int64_t> to_shape(const int64_t* dims, int64_t ndim) {
  return {dims, dims + ndim};
}

// Shared body of the four tr_from_data* entry points: validate numel
// against the shape and copy the host data into tensor-owned CPU storage
// of the given dtype. The caller applies the device move — the one place
// the *_on variants differ.
template <typename T>
torch::Tensor host_from_data(const T* data, uint64_t numel, const int64_t* dims,
                             int64_t ndim, torch::ScalarType dtype) {
  const auto shape = to_shape(dims, ndim);
  // Overflow-safe product: a crafted shape whose product wraps could
  // otherwise pass the numel check and hand from_blob an undersized
  // buffer.
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
  // Empty tensors carry no data pointer (an empty Racket vector marshals
  // as NULL): construct directly rather than handing from_blob a null.
  // torch.tensor([]) parity — dtype inference keeps these float32.
  if (numel == 0) {
    return torch::empty(shape, torch::TensorOptions().dtype(dtype));
  }
  // `data` is host memory, so from_blob must wrap it as a CPU tensor (a CUDA
  // default_options would make from_blob reject the host pointer); clone()
  // copies it into tensor-owned storage.
  return torch::from_blob(const_cast<T*>(data), shape,
                          torch::TensorOptions().dtype(dtype))
      .clone();
}

}  // namespace

extern "C" {

tr_tensor* tr_zeros(const int64_t* dims, int64_t ndim) {
  if (bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_zeros");
  }
  return torchrkt::alloc_result("tr_zeros", [&] {
    return torch::zeros(to_shape(dims, ndim), default_options());
  });
}

tr_tensor* tr_ones(const int64_t* dims, int64_t ndim) {
  if (bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_ones");
  }
  return torchrkt::alloc_result("tr_ones", [&] {
    return torch::ones(to_shape(dims, ndim), default_options());
  });
}

tr_tensor* tr_full(const int64_t* dims, int64_t ndim, double value) {
  if (bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_full");
  }
  return torchrkt::alloc_result("tr_full", [&] {
    return torch::full(to_shape(dims, ndim), value, default_options());
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

tr_tensor* tr_from_data_i64_on(const int64_t* data, uint64_t numel,
                               const int64_t* dims, int64_t ndim,
                               tr_device_type device_type,
                               int64_t device_index) {
  if ((!data && numel > 0) || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_i64_on");
  }
  return torchrkt::alloc_result("tr_from_data_i64_on", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kInt64)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

tr_tensor* tr_from_data_on(const float* data, uint64_t numel,
                           const int64_t* dims, int64_t ndim,
                           tr_device_type device_type, int64_t device_index) {
  if ((!data && numel > 0) || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data_on");
  }
  return torchrkt::alloc_result("tr_from_data_on", [&] {
    return host_from_data(data, numel, dims, ndim, torch::kFloat32)
        .to(torchrkt::to_torch_device(device_type, device_index));
  });
}

}  // extern "C"
