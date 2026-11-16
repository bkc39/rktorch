#include "torchrkt/c_api/random.h"

#include <ATen/CPUGeneratorImpl.h>
#include <torch/torch.h>

#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "torchrkt/detail/device.hpp"
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

int tr_tensor_uniform_gen_(tr_tensor* t, double low, double high,
                           tr_generator* g) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_uniform_gen_");
  }
  return torchrkt::status_call("tr_tensor_uniform_gen_", [&] {
    t->value.uniform_(low, high, generator_of(g));
  });
}

int tr_tensor_normal_(tr_tensor* t, double mean, double std, tr_generator* g) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_normal_");
  }
  return torchrkt::status_call("tr_tensor_normal_", [&] {
    t->value.normal_(mean, std, generator_of(g));
  });
}

tr_generator* tr_generator_new(uint64_t seed) {
  return torchrkt::alloc_handle<tr_generator>(
      "tr_generator_new", [&] { return at::detail::createCPUGenerator(seed); });
}

tr_generator* tr_generator_new_on(uint64_t seed, tr_device_type type,
                                  int64_t index) {
  return torchrkt::alloc_handle<tr_generator>("tr_generator_new_on", [&] {
    const torch::Device device = torchrkt::to_torch_device(type, index);
    if (device.is_cpu()) {
      return at::detail::createCPUGenerator(seed);
    }
    at::Generator g =
        at::globalContext()
            .getAcceleratorHooksInterface(device.type())
            .getNewGenerator(device.has_index() ? device.index() : 0);
    g.set_current_seed(seed);
    return g;
  });
}

int tr_generator_device(const tr_generator* g, tr_device_type* out_type,
                        int64_t* out_index) {
  if (!g || !out_type || !out_index) {
    return torchrkt::null_arg_status("tr_generator_device");
  }
  return torchrkt::status_call("tr_generator_device", [&] {
    const torch::Device d = g->value.device();
    *out_type = torchrkt::type_of(d);
    *out_index = d.has_index() ? static_cast<int64_t>(d.index()) : 0;
  });
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
    const torch::Device device = g ? g->value.device() : torch::kCPU;
    *out = torch::empty(
               {}, torch::TensorOptions().dtype(torch::kLong).device(device))
               .random_(generator_of(g))
               .item<int64_t>();
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

namespace {

torch::Tensor state_of(at::Generator gen) {
  const std::lock_guard<std::mutex> lock(gen.mutex());
  return gen.get_state();
}

// libtorch 2.14's CPU set_state reads from the start of the storage, so a
// valid state seen through a view at an offset is rejected: hand it a copy.
void set_state_of(at::Generator gen, const torch::Tensor& state) {
  const torch::Tensor copy = state.contiguous().clone();
  const std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_state(copy);
}

at::Generator default_generator(tr_device_type type, int64_t index) {
  return at::globalContext().defaultGenerator(
      torchrkt::to_torch_device(type, index));
}

}  // namespace

tr_tensor* tr_generator_state(const tr_generator* g) {
  if (!g) {
    return torchrkt::null_arg("tr_generator_state");
  }
  return torchrkt::alloc_result("tr_generator_state",
                                [&] { return state_of(g->value); });
}

int tr_generator_set_state(tr_generator* g, const tr_tensor* state) {
  if (!g || !state) {
    return torchrkt::null_arg_status("tr_generator_set_state");
  }
  return torchrkt::status_call("tr_generator_set_state",
                               [&] { set_state_of(g->value, state->value); });
}

tr_tensor* tr_default_generator_state(tr_device_type type, int64_t index) {
  return torchrkt::alloc_result("tr_default_generator_state", [&] {
    return state_of(default_generator(type, index));
  });
}

int tr_default_generator_set_state(tr_device_type type, int64_t index,
                                   const tr_tensor* state) {
  if (!state) {
    return torchrkt::null_arg_status("tr_default_generator_set_state");
  }
  return torchrkt::status_call("tr_default_generator_set_state", [&] {
    set_state_of(default_generator(type, index), state->value);
  });
}

}  // extern "C"
