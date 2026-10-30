#include "torchrkt/c_api/autograd.h"

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "torchrkt/detail/device.hpp"
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

int tr_tensor_has_grad(const tr_tensor* t, int* out) {
  if (!t || !out) {
    return torchrkt::null_arg_status("tr_tensor_has_grad");
  }
  return torchrkt::status_call(
      "tr_tensor_has_grad", [&] { *out = t->value.grad().defined() ? 1 : 0; });
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

int tr_tensors_reset_grad(tr_tensor* const* tensors, int64_t n) {
  if (torchrkt::tensors_arg_status("tr_tensors_reset_grad", tensors, n) != 0) {
    return 1;
  }
  return torchrkt::status_call("tr_tensors_reset_grad", [&] {
    for (int64_t i = 0; i < n; ++i) {
      tensors[i]->value.mutable_grad().reset();
    }
  });
}

int tr_tensors_grad(tr_tensor* const* tensors, int64_t n, tr_tensor** grads,
                    int64_t* nbytes, int* device_type, int64_t* device_index) {
  if (n < 0) {
    return torchrkt::tensors_arg_status("tr_tensors_grad", tensors, n);
  }
  if (n > 0 && grads) {
    std::fill(grads, grads + n, nullptr);
  }
  if (n > 0 && (!grads || !nbytes || !device_type || !device_index)) {
    return torchrkt::null_arg_status("tr_tensors_grad");
  }
  std::fill(nbytes, nbytes + n, 0);
  std::fill(device_type, device_type + n, static_cast<int>(TR_DEVICE_CPU));
  std::fill(device_index, device_index + n, 0);
  if (torchrkt::tensors_arg_status("tr_tensors_grad", tensors, n) != 0) {
    return 1;
  }
  std::vector<std::unique_ptr<tr_tensor>> held(static_cast<size_t>(n));
  const int rc = torchrkt::status_call("tr_tensors_grad", [&] {
    for (int64_t i = 0; i < n; ++i) {
      const torch::Tensor& g = tensors[i]->value.grad();
      if (!g.defined()) {
        continue;
      }
      const torch::Device d = g.device();
      if (!d.is_cpu() && !d.is_cuda() && !d.is_mps()) {
        throw std::invalid_argument(
            "gradient is on an unsupported device kind");
      }
      nbytes[i] = static_cast<int64_t>(g.nbytes());
      device_type[i] = static_cast<int>(torchrkt::type_of(d));
      device_index[i] = d.has_index() ? static_cast<int64_t>(d.index()) : 0;
      held[static_cast<size_t>(i)] = std::make_unique<tr_tensor>(tr_tensor{g});
    }
  });
  if (rc == 0) {
    for (int64_t i = 0; i < n; ++i) {
      grads[i] = held[static_cast<size_t>(i)].release();
    }
  }
  return rc;
}

void tr_tensors_free(tr_tensor* const* tensors, int64_t n) {
  if (!tensors) {
    return;
  }
  for (int64_t i = 0; i < n; ++i) {
    delete tensors[i];
  }
}

}  // extern "C"
