#include "torchrkt/c_api/shape_ops.h"

#include <torch/torch.h>

#include <vector>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

bool bad_dims(const int64_t* dims, int64_t ndim) {
  return ndim < 0 || (ndim > 0 && !dims);
}

std::vector<int64_t> to_shape(const int64_t* dims, int64_t ndim) {
  return {dims, dims + ndim};
}

std::vector<torch::Tensor> collect(const tr_tensor* const* tensors, int64_t n) {
  std::vector<torch::Tensor> values;
  values.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    if (!tensors[i]) {
      throw std::invalid_argument("NULL tensor in list");
    }
    values.push_back(tensors[i]->value);
  }
  return values;
}

}  // namespace

extern "C" {

tr_tensor* tr_reshape(const tr_tensor* t, const int64_t* dims, int64_t ndim) {
  if (!t || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_reshape");
  }
  return torchrkt::alloc_result(
      "tr_reshape", [&] { return t->value.reshape(to_shape(dims, ndim)); });
}

tr_tensor* tr_view(const tr_tensor* t, const int64_t* dims, int64_t ndim) {
  if (!t || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_view");
  }
  return torchrkt::alloc_result(
      "tr_view", [&] { return t->value.view(to_shape(dims, ndim)); });
}

tr_tensor* tr_transpose(const tr_tensor* t, int64_t dim0, int64_t dim1) {
  if (!t) {
    return torchrkt::null_arg("tr_transpose");
  }
  return torchrkt::alloc_result("tr_transpose",
                                [&] { return t->value.transpose(dim0, dim1); });
}

tr_tensor* tr_permute(const tr_tensor* t, const int64_t* dims, int64_t ndim) {
  if (!t || bad_dims(dims, ndim)) {
    return torchrkt::null_arg("tr_permute");
  }
  return torchrkt::alloc_result(
      "tr_permute", [&] { return t->value.permute(to_shape(dims, ndim)); });
}

tr_tensor* tr_squeeze(const tr_tensor* t) {
  if (!t) {
    return torchrkt::null_arg("tr_squeeze");
  }
  return torchrkt::alloc_result("tr_squeeze",
                                [&] { return t->value.squeeze(); });
}

tr_tensor* tr_squeeze_dim(const tr_tensor* t, int64_t dim) {
  if (!t) {
    return torchrkt::null_arg("tr_squeeze_dim");
  }
  return torchrkt::alloc_result("tr_squeeze_dim",
                                [&] { return t->value.squeeze(dim); });
}

tr_tensor* tr_unsqueeze(const tr_tensor* t, int64_t dim) {
  if (!t) {
    return torchrkt::null_arg("tr_unsqueeze");
  }
  return torchrkt::alloc_result("tr_unsqueeze",
                                [&] { return t->value.unsqueeze(dim); });
}

tr_tensor* tr_cat(const tr_tensor* const* tensors, int64_t n, int64_t dim) {
  if (!tensors || n <= 0) {
    return torchrkt::null_arg("tr_cat");
  }
  return torchrkt::alloc_result(
      "tr_cat", [&] { return torch::cat(collect(tensors, n), dim); });
}

tr_tensor* tr_stack(const tr_tensor* const* tensors, int64_t n, int64_t dim) {
  if (!tensors || n <= 0) {
    return torchrkt::null_arg("tr_stack");
  }
  return torchrkt::alloc_result(
      "tr_stack", [&] { return torch::stack(collect(tensors, n), dim); });
}

}  // extern "C"
