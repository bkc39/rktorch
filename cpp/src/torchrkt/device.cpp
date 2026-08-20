#include "torchrkt/c_api/device.h"

#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace torchrkt {

namespace {

// The default device, packed lock-free: the CUDA bit in bit 0, the device
// ordinal above it. The zero state is CPU index 0, matching the v1 default, so
// a never-set process behaves exactly as before. seq_cst (not relaxed) so a
// store in one Racket place publishes to a load in another (e.g. ARM64, where
// relaxed carries no cross-thread happens-before).
std::atomic<int64_t> g_default_device{0};

int64_t pack_device(tr_device_type type, int64_t index) {
  return (index << 1) | (type == TR_DEVICE_CUDA ? 1 : 0);
}

}  // namespace

torch::Device to_torch_device(tr_device_type type, int64_t index) {
  switch (type) {
    case TR_DEVICE_CPU:
      return torch::Device(torch::kCPU);
    case TR_DEVICE_CUDA:
      // Range-check BEFORE narrowing: torch::DeviceIndex is 8-bit, so an
      // unvalidated ordinal like 256 would silently wrap to device 0 and
      // place tensors on the wrong GPU instead of erroring.
      if (index < 0 ||
          index >= static_cast<int64_t>(torch::cuda::device_count())) {
        throw std::invalid_argument("CUDA device index out of range");
      }
      return torch::Device(torch::kCUDA,
                           static_cast<torch::DeviceIndex>(index));
  }
  throw std::invalid_argument("unknown tr_device_type");
}

torch::Device current_default_device() {
  const int64_t packed = g_default_device.load(std::memory_order_seq_cst);
  const tr_device_type type =
      (packed & 1) != 0 ? TR_DEVICE_CUDA : TR_DEVICE_CPU;
  // Unpack via an unsigned intermediate: right-shifting a negative signed
  // int64_t is implementation-defined, and pack_device carries no guard of its
  // own that the high bit is clear.
  const int64_t index =
      static_cast<int64_t>(static_cast<uint64_t>(packed) >> 1U);
  return to_torch_device(type, index);
}

void set_default_device(tr_device_type type, int64_t index) {
  if (type == TR_DEVICE_CUDA) {
    if (!torch::cuda::is_available()) {
      throw std::invalid_argument("CUDA is not available");
    }
    if (index < 0 ||
        index >= static_cast<int64_t>(torch::cuda::device_count())) {
      throw std::invalid_argument("CUDA device index out of range");
    }
  } else if (type == TR_DEVICE_CPU) {
    // CPU has no ordinal: a non-zero index would overflow pack_device's shift
    // and fail to round-trip (torch::Device(kCPU) carries no index), so reject
    // it rather than silently accept-and-lose it.
    if (index != 0) {
      throw std::invalid_argument("CPU device index must be 0");
    }
  } else {
    throw std::invalid_argument("unknown tr_device_type");
  }
  g_default_device.store(pack_device(type, index), std::memory_order_seq_cst);
}

}  // namespace torchrkt

extern "C" {

// torch::cuda::is_available / device_count are not documented noexcept (a
// driver/CUDA-init failure can throw), and these return result values, not the
// int-status the op_call.hpp helpers expect, so they catch all and return 0.
// They still record the message in tr_last_error (like alloc_result), so a
// driver-init failure is distinguishable from "genuinely no CUDA".
int tr_cuda_is_available(void) {
  try {
    return torch::cuda::is_available() ? 1 : 0;
  } catch (const std::exception& e) {
    torchrkt::set_error(std::string("tr_cuda_is_available: ") + e.what(),
                        torchrkt::classify(e));
    return 0;
  } catch (...) {
    torchrkt::set_error("tr_cuda_is_available: unknown exception");
    return 0;
  }
}

int tr_cuda_device_count(void) {
  try {
    return torch::cuda::is_available()
               ? static_cast<int>(torch::cuda::device_count())
               : 0;
  } catch (const std::exception& e) {
    torchrkt::set_error(std::string("tr_cuda_device_count: ") + e.what(),
                        torchrkt::classify(e));
    return 0;
  } catch (...) {
    torchrkt::set_error("tr_cuda_device_count: unknown exception");
    return 0;
  }
}

int tr_set_default_device(tr_device_type type, int64_t index) {
  return torchrkt::status_call("tr_set_default_device", [&] {
    torchrkt::set_default_device(type, index);
  });
}

int tr_get_default_device(tr_device_type* out_type, int64_t* out_index) {
  if (!out_type || !out_index) {
    return torchrkt::null_arg_status("tr_get_default_device");
  }
  return torchrkt::status_call("tr_get_default_device", [&] {
    const torch::Device d = torchrkt::current_default_device();
    *out_type = d.is_cuda() ? TR_DEVICE_CUDA : TR_DEVICE_CPU;
    *out_index = d.has_index() ? static_cast<int64_t>(d.index()) : 0;
  });
}

tr_tensor* tr_tensor_to_device(const tr_tensor* t, tr_device_type type,
                               int64_t index) {
  if (!t) {
    return torchrkt::null_arg("tr_tensor_to_device");
  }
  return torchrkt::alloc_result("tr_tensor_to_device", [&] {
    return t->value.to(torchrkt::to_torch_device(type, index));
  });
}

int tr_tensor_device(const tr_tensor* t, tr_device_type* out_type,
                     int64_t* out_index) {
  if (!t || !out_type || !out_index) {
    return torchrkt::null_arg_status("tr_tensor_device");
  }
  return torchrkt::status_call("tr_tensor_device", [&] {
    const torch::Device d = t->value.device();
    // Reject device kinds outside the C ABI (e.g. a future MPS/XPU tensor, see
    // #13) rather than silently labelling them CPU. (tr_get_default_device
    // needs no such guard: current_default_device only ever yields CPU/CUDA.)
    if (!d.is_cpu() && !d.is_cuda()) {
      throw std::invalid_argument("tensor is on an unsupported device kind");
    }
    *out_type = d.is_cuda() ? TR_DEVICE_CUDA : TR_DEVICE_CPU;
    *out_index = d.has_index() ? static_cast<int64_t>(d.index()) : 0;
  });
}

}  // extern "C"
