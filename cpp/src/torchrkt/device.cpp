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

#ifdef TORCHRKT_WITH_CUDA_ALLOCATOR
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace torchrkt {

namespace {

// Packed lock-free: CUDA bit in bit 0, ordinal above it; zero = CPU:0.
// seq_cst so a store in one Racket place publishes to loads in another.
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
      // torch::DeviceIndex is 8-bit: range-check before narrowing, or an
      // ordinal like 256 silently wraps to device 0.
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
  // Unsigned intermediate: right-shifting a negative int64_t is
  // implementation-defined.
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
    // torch::Device(kCPU) carries no index, so a nonzero ordinal would not
    // round-trip: reject rather than silently lose it.
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

// torch::cuda::is_available / device_count can throw on driver-init failure
// and return values, not statuses, so they hand-roll the catch: return 0 but
// still record in tr_last_error, distinguishing "CUDA broke" from "no CUDA".
int tr_cuda_is_available(void) {
  try {
    return torch::cuda::is_available() ? 1 : 0;
  } catch (const std::exception& e) {
    torchrkt::record_failure("tr_cuda_is_available", e);
    return 0;
  } catch (...) {
    torchrkt::record_unknown_failure("tr_cuda_is_available");
    return 0;
  }
}

int tr_cuda_device_count(void) {
  try {
    return torch::cuda::is_available()
               ? static_cast<int>(torch::cuda::device_count())
               : 0;
  } catch (const std::exception& e) {
    torchrkt::record_failure("tr_cuda_device_count", e);
    return 0;
  } catch (...) {
    torchrkt::record_unknown_failure("tr_cuda_device_count");
    return 0;
  }
}

int tr_cuda_memory_stats(int64_t device_index, int64_t* out_allocated,
                         int64_t* out_reserved, int64_t* out_peak_allocated) {
  if (!out_allocated || !out_reserved || !out_peak_allocated) {
    return torchrkt::null_arg_status("tr_cuda_memory_stats");
  }
#ifndef TORCHRKT_WITH_CUDA_ALLOCATOR
  (void)device_index;
  // Throw through status_call so the recording rides the noexcept-safe path
  // (a direct set_error would allocate outside any catch).
  return torchrkt::status_call("tr_cuda_memory_stats", [] {
    throw std::runtime_error("CUDA support is not compiled into this build");
  });
#else
  return torchrkt::status_call("tr_cuda_memory_stats", [&] {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA is not available");
    }
    if (device_index < 0 ||
        device_index >= static_cast<int64_t>(torch::cuda::device_count())) {
      throw std::invalid_argument("CUDA device index out of range");
    }
    // A never-initialized allocator reports zeros, matching
    // torch.cuda.memory_allocated() before first use; probe explicitly —
    // catching getDeviceStats' throw would mask real failures as zeros.
    if (!c10::cuda::CUDACachingAllocator::get()->initialized()) {
      *out_allocated = 0;
      *out_reserved = 0;
      *out_peak_allocated = 0;
      return;
    }
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        static_cast<c10::DeviceIndex>(device_index));
    const auto agg =
        static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
    *out_allocated = stats.allocated_bytes[agg].current;
    *out_reserved = stats.reserved_bytes[agg].current;
    *out_peak_allocated = stats.allocated_bytes[agg].peak;
  });
#endif
}

int tr_cuda_empty_cache(void) {
  return torchrkt::status_call("tr_cuda_empty_cache", [&] {
#ifdef TORCHRKT_WITH_CUDA_ALLOCATOR
    // The no-CUDA and CPU-build paths are deliberate no-op successes so the
    // OOM retry can call this unconditionally.
    if (torch::cuda::is_available()) {
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
  });
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
    // Reject device kinds outside the C ABI (a future MPS/XPU tensor) rather
    // than silently labelling them CPU.
    if (!d.is_cpu() && !d.is_cuda()) {
      throw std::invalid_argument("tensor is on an unsupported device kind");
    }
    *out_type = d.is_cuda() ? TR_DEVICE_CUDA : TR_DEVICE_CPU;
    *out_index = d.has_index() ? static_cast<int64_t>(d.index()) : 0;
  });
}

}  // extern "C"
