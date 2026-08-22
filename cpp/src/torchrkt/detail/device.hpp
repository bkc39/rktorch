#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "torchrkt/c_api/device.h"

namespace torchrkt {

torch::Device to_torch_device(tr_device_type type, int64_t index);

torch::Device current_default_device();

void set_default_device(tr_device_type type, int64_t index);

}  // namespace torchrkt
