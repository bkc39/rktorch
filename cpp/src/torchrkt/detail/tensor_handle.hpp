#pragma once

#include <torch/torch.h>

struct tr_tensor {
  torch::Tensor value;
};
