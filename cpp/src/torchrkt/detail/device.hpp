#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"

namespace torchrkt {

torch::Device to_torch_device(tr_device_type type, int64_t index);

torch::Device current_default_device();

void set_default_device(tr_device_type type, int64_t index);

// Tensor::to over optional device/dtype: the KEEP sentinels leave an axis
// unchanged; copy=false, so an unchanged tensor comes back aliased.
torch::Tensor convert(const torch::Tensor& v, tr_device_type type,
                      int64_t index, tr_dtype dtype);

bool rebindable(const torch::Tensor& dst, const torch::Tensor& src);

}  // namespace torchrkt
