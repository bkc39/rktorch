#include "torchrkt/c_api/tensor.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>

#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

torch::ScalarType to_scalar_type(tr_dtype dtype) {
  switch (dtype) {
    case TR_DTYPE_FLOAT32:
      return torch::kFloat32;
    case TR_DTYPE_FLOAT64:
      return torch::kFloat64;
    case TR_DTYPE_INT64:
      return torch::kInt64;
    case TR_DTYPE_BOOL:
      return torch::kBool;
  }
  throw std::invalid_argument("unknown tr_dtype");
}

}  // namespace

extern "C" {

void tr_tensor_free(tr_tensor* t) {
  // Racket GC finalizer; deliberately NO try/catch. A throw during storage
  // release unwinds through libtorch's implicitly-noexcept frames and reaches
  // std::terminate before any handler here (pinned by
  // finalizer_death_test.cpp); the safety guarantee is the Racket-side
  // deallocator wrap in raw/memory.rkt.
  delete t;
}

int tr_tensor_numel(const tr_tensor* t, int64_t* out) {
  if (!t || !out) {
    torchrkt::set_error("tr_tensor_numel: null argument");
    return 1;
  }
  try {
    *out = t->value.numel();
    return 0;
  } catch (const std::exception& e) {
    torchrkt::record_failure("tr_tensor_numel", e);
    return 1;
  }
}

int tr_tensor_nbytes(const tr_tensor* t, int64_t* out) {
  if (!t || !out) {
    return torchrkt::null_arg_status("tr_tensor_nbytes");
  }
  // The view's extent, not the shared storage's: the Racket memory ledger
  // charges each handle for what it addresses.
  return torchrkt::status_call("tr_tensor_nbytes", [&] {
    *out = static_cast<int64_t>(t->value.nbytes());
  });
}

int tr_tensor_ndim(const tr_tensor* t, int64_t* out) {
  if (!t || !out) {
    torchrkt::set_error("tr_tensor_ndim: null argument");
    return 1;
  }
  try {
    *out = t->value.dim();
    return 0;
  } catch (const std::exception& e) {
    torchrkt::record_failure("tr_tensor_ndim", e);
    return 1;
  }
}

int tr_tensor_shape(const tr_tensor* t, int64_t capacity, int64_t* out_dims,
                    int64_t* out_ndim) {
  if (!t || !out_ndim) {
    torchrkt::set_error("tr_tensor_shape: null argument");
    return 1;
  }
  *out_ndim = 0;
  try {
    const auto sizes = t->value.sizes();
    *out_ndim = static_cast<int64_t>(sizes.size());
    if (capacity < *out_ndim) {
      return 2;
    }
    if (out_dims) {
      for (int64_t i = 0; i < *out_ndim; ++i) {
        out_dims[i] = sizes[static_cast<size_t>(i)];
      }
    }
    return 0;
  } catch (const std::exception& e) {
    torchrkt::record_failure("tr_tensor_shape", e);
    return 1;
  }
}

int tr_tensor_dtype(const tr_tensor* t, tr_dtype* out) {
  if (!t || !out) {
    return torchrkt::null_arg_status("tr_tensor_dtype");
  }
  return torchrkt::status_call("tr_tensor_dtype", [&] {
    switch (t->value.scalar_type()) {
      case torch::kFloat32:
        *out = TR_DTYPE_FLOAT32;
        return;
      case torch::kFloat64:
        *out = TR_DTYPE_FLOAT64;
        return;
      case torch::kInt64:
        *out = TR_DTYPE_INT64;
        return;
      case torch::kBool:
        *out = TR_DTYPE_BOOL;
        return;
      default:
        // Reject dtypes outside the C ABI rather than mislabel them.
        throw std::invalid_argument("tensor has an unsupported dtype");
    }
  });
}

int tr_tensor_copy_data_i64(const tr_tensor* t, uint64_t capacity, int64_t* out,
                            uint64_t* out_numel) {
  if (!t || !out_numel) {
    return torchrkt::null_arg_status("tr_tensor_copy_data_i64");
  }
  return torchrkt::copy_data_call(
      "tr_tensor_copy_data_i64", capacity, out, out_numel,
      [&] { return t->value.to(torch::kCPU).to(torch::kInt64).contiguous(); });
}

int tr_tensor_copy_data_f64(const tr_tensor* t, uint64_t capacity, double* out,
                            uint64_t* out_numel) {
  if (!t || !out_numel) {
    return torchrkt::null_arg_status("tr_tensor_copy_data_f64");
  }
  return torchrkt::copy_data_call(
      "tr_tensor_copy_data_f64", capacity, out, out_numel, [&] {
        return t->value.to(torch::kCPU).to(torch::kFloat64).contiguous();
      });
}

int tr_tensor_copy_data(const tr_tensor* t, uint64_t capacity, float* out,
                        uint64_t* out_numel) {
  if (!t || !out_numel) {
    return torchrkt::null_arg_status("tr_tensor_copy_data");
  }
  return torchrkt::copy_data_call(
      "tr_tensor_copy_data", capacity, out, out_numel, [&] {
        return t->value.to(torch::kCPU).to(torch::kFloat32).contiguous();
      });
}

int tr_tensor_item(const tr_tensor* t, double* out) {
  if (!t || !out) {
    return torchrkt::null_arg_status("tr_tensor_item");
  }
  return torchrkt::status_call("tr_tensor_item",
                               [&] { *out = t->value.item<double>(); });
}

tr_tensor* tr_tensor_to_dtype(const tr_tensor* t, tr_dtype dtype) {
  if (!t) {
    return torchrkt::null_arg("tr_tensor_to_dtype");
  }
  return torchrkt::alloc_result(
      "tr_tensor_to_dtype", [&] { return t->value.to(to_scalar_type(dtype)); });
}

int tr_tensor_print(const tr_tensor* t, uint64_t buffer_capacity,
                    char* out_buffer, uint64_t* out_len) {
  if (!t || !out_len) {
    torchrkt::set_error("tr_tensor_print: null argument");
    return 1;
  }
  *out_len = 0;
  try {
    std::ostringstream os;
    os << t->value;
    const std::string rendered = os.str();
    const uint64_t len = static_cast<uint64_t>(rendered.size());
    *out_len = len;
    if (buffer_capacity < len) {
      return 2;
    }
    if (out_buffer && len > 0) {
      // std::copy (not memcpy) keeps clang-tidy's not-null-terminated check
      // quiet: the size-then-fill contract deliberately omits the NUL.
      std::copy(rendered.data(), rendered.data() + len, out_buffer);
    }
    return 0;
  } catch (const std::exception& e) {
    torchrkt::record_failure("tr_tensor_print", e);
    return 1;
  }
}

}  // extern "C"
