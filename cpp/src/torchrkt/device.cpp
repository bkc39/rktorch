#include "torchrkt/c_api/device.h"

#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace torchrkt {

namespace {

// The default device, packed so reads stay lock-free: the CUDA bit in bit 0,
// the device ordinal above it. The zero state is CPU index 0, matching the v1
// default, so a never-set process behaves exactly as before.
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
      return torch::Device(torch::kCUDA,
                           static_cast<torch::DeviceIndex>(index));
  }
  throw std::invalid_argument("unknown tr_device_type");
}

torch::Device current_default_device() {
  const int64_t packed = g_default_device.load(std::memory_order_relaxed);
  const tr_device_type type =
      (packed & 1) != 0 ? TR_DEVICE_CUDA : TR_DEVICE_CPU;
  return to_torch_device(type, packed >> 1);
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
  } else if (type != TR_DEVICE_CPU) {
    throw std::invalid_argument("unknown tr_device_type");
  }
  g_default_device.store(pack_device(type, index), std::memory_order_relaxed);
}

}  // namespace torchrkt

extern "C" {

int tr_cuda_is_available(void) {
  return torch::cuda::is_available() ? 1 : 0;
}

int tr_cuda_device_count(void) {
  if (!torch::cuda::is_available()) {
    return 0;
  }
  return static_cast<int>(torch::cuda::device_count());
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
    *out_type = d.is_cuda() ? TR_DEVICE_CUDA : TR_DEVICE_CPU;
    *out_index = d.has_index() ? static_cast<int64_t>(d.index()) : 0;
  });
}

}  // extern "C"
