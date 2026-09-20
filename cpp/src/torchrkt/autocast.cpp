#include "torchrkt/c_api/autocast.h"

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include <stdexcept>

#include "torchrkt/detail/dtype.hpp"
#include "torchrkt/detail/op_call.hpp"

namespace {

torch::DeviceType device_type_of(tr_device_type type) {
  switch (type) {
    case TR_DEVICE_CPU:
      return torch::kCPU;
    case TR_DEVICE_CUDA:
      return torch::kCUDA;
    case TR_DEVICE_MPS:
      return torch::kMPS;
    case TR_DEVICE_KEEP:
      break;
  }
  throw std::invalid_argument("unknown tr_device_type");
}

tr_dtype half_dtype_of(torch::ScalarType scalar) {
  switch (scalar) {
    case torch::kFloat16:
      return TR_DTYPE_FLOAT16;
    case torch::kBFloat16:
      return TR_DTYPE_BFLOAT16;
    default:
      throw std::invalid_argument("autocast dtype outside the half pair");
  }
}

}  // namespace

extern "C" {

int tr_set_autocast_enabled(tr_device_type type, tr_dtype dtype, int enabled) {
  return torchrkt::status_call("tr_set_autocast_enabled", [&] {
    const torch::DeviceType device = device_type_of(type);
    // the dtype is restored on the way out whether or not autocast was on
    // there, so a body that changed it does not outlive its extent
    if (dtype != TR_DTYPE_KEEP) {
      if (dtype != TR_DTYPE_FLOAT16 && dtype != TR_DTYPE_BFLOAT16) {
        throw std::invalid_argument(
            "autocast dtype must be float16 or bfloat16");
      }
      at::autocast::set_autocast_dtype(device, torchrkt::to_scalar_type(dtype));
    } else if (enabled != 0) {
      throw std::invalid_argument("autocast dtype must be float16 or bfloat16");
    }
    if (enabled == 0) {
      at::autocast::set_autocast_enabled(device, false);
      at::autocast::clear_cache();
      return;
    }
    at::autocast::set_autocast_enabled(device, true);
  });
}

int tr_is_autocast_enabled(tr_device_type type, int* out) {
  if (!out) {
    return torchrkt::null_arg_status("tr_is_autocast_enabled");
  }
  return torchrkt::status_call("tr_is_autocast_enabled", [&] {
    *out = at::autocast::is_autocast_enabled(device_type_of(type)) ? 1 : 0;
  });
}

int tr_autocast_dtype(tr_device_type type, tr_dtype* out) {
  if (!out) {
    return torchrkt::null_arg_status("tr_autocast_dtype");
  }
  return torchrkt::status_call("tr_autocast_dtype", [&] {
    *out =
        half_dtype_of(at::autocast::get_autocast_dtype(device_type_of(type)));
  });
}

}  // extern "C"
