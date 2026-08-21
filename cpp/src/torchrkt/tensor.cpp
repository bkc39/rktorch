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
  }
  throw std::invalid_argument("unknown tr_dtype");
}

}  // namespace

extern "C" {

void tr_tensor_free(tr_tensor* t) {
  // Called from Racket GC finalizers. Deliberately NO try/catch: none can
  // work here. A throw during storage release (the CUDA caching allocator
  // failing on an errored context) unwinds through libtorch's own
  // implicitly-noexcept frames — TensorBase's noexcept move-assign,
  // ~TensorImpl/~StorageImpl/~DataPtr — and reaches std::terminate before
  // any handler at this layer, wherever the release statement is placed.
  // Pinned empirically by finalizer_death_test.cpp (a catch-based version
  // of this function still aborted the child). The finalizer-safety
  // guarantee lives on the Racket side instead: raw/memory.rkt wraps this
  // binding's deallocator and swallows the runtime-converted failure class
  // actually observed in issue #38.
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
  // The view's extent (numel x element size), not the shared storage's:
  // the Racket-side memory ledger charges each handle for what it
  // addresses, a documented approximation for GC pressure (#37).
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
      default:
        // Reject rather than mislabel (the tr_tensor_device pattern for
        // kinds outside the C ABI).
        throw std::invalid_argument("tensor has an unsupported dtype");
    }
  });
}

int tr_tensor_copy_data_i64(const tr_tensor* t, uint64_t capacity, int64_t* out,
                            uint64_t* out_numel) {
  if (!t || !out_numel) {
    torchrkt::set_error("tr_tensor_copy_data_i64: null argument");
    return 1;
  }
  *out_numel = 0;
  try {
    const torch::Tensor c =
        t->value.to(torch::kCPU).to(torch::kInt64).contiguous();
    const uint64_t numel = static_cast<uint64_t>(c.numel());
    *out_numel = numel;
    if (capacity < numel) {
      return 2;
    }
    if (out && numel > 0) {
      std::memcpy(out, c.data_ptr<int64_t>(), numel * sizeof(int64_t));
    }
    return 0;
  } catch (const std::exception& e) {
    // Same noexcept-safe recording as tr_tensor_copy_data: the
    // contiguous copy can be large.
    torchrkt::record_failure("tr_tensor_copy_data_i64", e);
    return 1;
  }
}

int tr_tensor_copy_data(const tr_tensor* t, uint64_t capacity, float* out,
                        uint64_t* out_numel) {
  if (!t || !out_numel) {
    torchrkt::set_error("tr_tensor_copy_data: null argument");
    return 1;
  }
  *out_numel = 0;
  try {
    const torch::Tensor c =
        t->value.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const uint64_t numel = static_cast<uint64_t>(c.numel());
    *out_numel = numel;
    if (capacity < numel) {
      return 2;
    }
    if (out && numel > 0) {
      std::memcpy(out, c.data_ptr<float>(), numel * sizeof(float));
    }
    return 0;
  } catch (const std::exception& e) {
    // The contiguous float copy above can be large: record noexcept-safely
    // (rich message when it can allocate, kind-only when it can't).
    torchrkt::record_failure("tr_tensor_copy_data", e);
    return 1;
  }
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
    // The render allocates the whole text: same noexcept-safe recording
    // as tr_tensor_copy_data.
    torchrkt::record_failure("tr_tensor_print", e);
    return 1;
  }
}

}  // extern "C"
