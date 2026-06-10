#include "torchrkt/c_api/autograd.h"

#include <torch/torch.h>

#include <stdexcept>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

extern "C" {

int tr_tensor_requires_grad_(tr_tensor* t, int requires_grad) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_requires_grad_");
  }
  return torchrkt::status_call("tr_tensor_requires_grad_", [&] {
    t->value.requires_grad_(requires_grad != 0);
  });
}

int tr_tensor_requires_grad(const tr_tensor* t, int* out) {
  if (!t || !out) {
    return torchrkt::null_arg_status("tr_tensor_requires_grad");
  }
  return torchrkt::status_call("tr_tensor_requires_grad", [&] {
    *out = t->value.requires_grad() ? 1 : 0;
  });
}

int tr_tensor_backward(tr_tensor* t) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_backward");
  }
  return torchrkt::status_call("tr_tensor_backward",
                               [&] { t->value.backward(); });
}

tr_tensor* tr_tensor_grad(const tr_tensor* t) {
  if (!t) {
    return torchrkt::null_arg("tr_tensor_grad");
  }
  return torchrkt::alloc_result("tr_tensor_grad", [&] {
    torch::Tensor g = t->value.grad();
    if (!g.defined()) {
      throw std::runtime_error(
          "tensor has no gradient (did backward run, and is this a leaf "
          "with requires_grad?)");
    }
    return g;
  });
}

tr_tensor* tr_tensor_detach(const tr_tensor* t) {
  if (!t) {
    return torchrkt::null_arg("tr_tensor_detach");
  }
  return torchrkt::alloc_result("tr_tensor_detach",
                                [&] { return t->value.detach(); });
}

int tr_set_grad_enabled(int enabled) {
  return torchrkt::status_call(
      "tr_set_grad_enabled", [&] { c10::GradMode::set_enabled(enabled != 0); });
}

int tr_is_grad_enabled(int* out) {
  if (!out) {
    return torchrkt::null_arg_status("tr_is_grad_enabled");
  }
  return torchrkt::status_call("tr_is_grad_enabled", [&] {
    *out = c10::GradMode::is_enabled() ? 1 : 0;
  });
}

int tr_tensor_sub_(tr_tensor* t, const tr_tensor* other, double alpha) {
  if (!t || !other) {
    return torchrkt::null_arg_status("tr_tensor_sub_");
  }
  return torchrkt::status_call("tr_tensor_sub_",
                               [&] { t->value.sub_(other->value, alpha); });
}

int tr_tensor_zero_(tr_tensor* t) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_zero_");
  }
  return torchrkt::status_call("tr_tensor_zero_", [&] { t->value.zero_(); });
}

int tr_tensor_mul_(tr_tensor* t, double value) {
  if (!t) {
    return torchrkt::null_arg_status("tr_tensor_mul_");
  }
  return torchrkt::status_call("tr_tensor_mul_", [&] { t->value.mul_(value); });
}

}  // extern "C"
