#pragma once

#include <torch/torch.h>

// Definition of the opaque handle declared in torchrkt/c_api/tensor.h. Kept in
// src/ (a PRIVATE include dir) so it never leaks into the public C surface.
struct tr_tensor {
  torch::Tensor value;
};
