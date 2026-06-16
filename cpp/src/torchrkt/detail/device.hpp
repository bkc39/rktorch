#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "torchrkt/c_api/device.h"

// Device translation shared by the creation/random constructors (which read the
// process-wide default device) and device.cpp (which owns it). Kept in src/ (a
// PRIVATE include dir) so it never leaks into the public C surface.

namespace torchrkt {

// Translate the C ABI descriptor into an ATen device. Throws
// std::invalid_argument on an unknown type so the boundary helpers turn it into
// a status/last-error rather than letting it cross the FFI line.
torch::Device to_torch_device(tr_device_type type, int64_t index);

// The device tr_zeros/tr_randn/... place new tensors on. Defaults to CPU; set
// via set_default_device. Reads are lock-free.
torch::Device current_default_device();

// Backs tr_set_default_device: validates CUDA availability and the ordinal,
// then publishes the new default. Throws on an invalid request.
void set_default_device(tr_device_type type, int64_t index);

}  // namespace torchrkt
