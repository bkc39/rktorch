#include "torchrkt/c_api/random.h"

#include <ATen/CPUGeneratorImpl.h>
#include <torch/torch.h>

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/generator_handle.hpp"
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

namespace {

std::optional<at::Generator> generator_of(tr_generator* g) {
  if (g) {
    return g->value;
  }
  return std::nullopt;
}

tr_tensor* drawn_on(const char* who, bool normal, const int64_t* dims,
                    int64_t ndim, tr_device_type type, int64_t index,
                    tr_dtype dtype, tr_generator* g) {
  return torchrkt::shaped_result(
      who, dims, ndim, [&](const std::vector<int64_t>& shape) {
        const auto options = torchrkt::options_on(type, index, dtype);
        return normal ? torch::randn(shape, generator_of(g), options)
                      : torch::rand(shape, generator_of(g), options);
      });
}

}  // namespace

tr_tensor* tr_randn_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                       int64_t index, tr_dtype dtype) {
  return drawn_on("tr_randn_on", true, dims, ndim, type, index, dtype, nullptr);
}

tr_tensor* tr_rand_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                      int64_t index, tr_dtype dtype) {
  return drawn_on("tr_rand_on", false, dims, ndim, type, index, dtype, nullptr);
}

tr_tensor* tr_randn_gen_on(const int64_t* dims, int64_t ndim,
                           tr_device_type type, int64_t index, tr_dtype dtype,
                           tr_generator* g) {
  return drawn_on("tr_randn_gen_on", true, dims, ndim, type, index, dtype, g);
}

tr_tensor* tr_rand_gen_on(const int64_t* dims, int64_t ndim,
                          tr_device_type type, int64_t index, tr_dtype dtype,
                          tr_generator* g) {
  return drawn_on("tr_rand_gen_on", false, dims, ndim, type, index, dtype, g);
}

int tr_tensor_uniform_(tr_tensor* t, double low, double high) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_uniform_");
  }
  return torchrkt::status_call("tr_tensor_uniform_",
                               [&] { t->value.uniform_(low, high); });
}

tr_generator* tr_generator_new(uint64_t seed) {
  return torchrkt::alloc_handle<tr_generator>(
      "tr_generator_new", [&] { return at::detail::createCPUGenerator(seed); });
}

void tr_generator_free(tr_generator* g) {
  // GC finalizer, as tr_tensor_free: deliberately NO try/catch, a throw
  // terminates inside libtorch's noexcept release first.
  delete g;
}

int tr_generator_draw_seed(tr_generator* g, int64_t* out) {
  if (!out) {
    return torchrkt::null_arg_status("tr_generator_draw_seed");
  }
  return torchrkt::status_call("tr_generator_draw_seed", [&] {
    *out =
        torch::empty({}, torch::kLong).random_(generator_of(g)).item<int64_t>();
  });
}

tr_tensor* tr_randperm(int64_t n, tr_generator* g) {
  return torchrkt::alloc_result("tr_randperm", [&] {
    if (n < 0) {
      throw std::invalid_argument("randperm: n must be non-negative");
    }
    return at::randperm(n, generator_of(g),
                        at::TensorOptions().dtype(at::kLong));
  });
}

}  // extern "C"
