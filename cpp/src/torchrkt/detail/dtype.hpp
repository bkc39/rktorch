#pragma once

#include <torch/torch.h>

#include <stdexcept>

#include "torchrkt/c_api/tensor.h"

namespace torchrkt {

inline torch::ScalarType to_scalar_type(tr_dtype dtype) {
  switch (dtype) {
    case TR_DTYPE_FLOAT32:
      return torch::kFloat32;
    case TR_DTYPE_FLOAT64:
      return torch::kFloat64;
    case TR_DTYPE_INT64:
      return torch::kInt64;
    case TR_DTYPE_BOOL:
      return torch::kBool;
    case TR_DTYPE_KEEP:
      break;
  }
  throw std::invalid_argument("unknown tr_dtype");
}

}  // namespace torchrkt
