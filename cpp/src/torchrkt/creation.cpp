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
  if (!data || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_from_data");
  }
  return torchrkt::alloc_result("tr_from_data", [&] {
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
    // from_blob borrows `data`; clone() copies it into tensor-owned storage.
    return torch::from_blob(const_cast<float*>(data), shape, default_options())
        .clone();
  });
}

}  // extern "C"
