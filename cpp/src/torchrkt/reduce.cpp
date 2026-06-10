#include "torchrkt/c_api/reduce.h"

#include <torch/torch.h>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

#define TR_UNARY_OP(name, expr)                                   \
  tr_tensor* name(const tr_tensor* t) {                           \
    if (!t) {                                                     \
      return torchrkt::null_arg(#name);                           \
    }                                                             \
    return torchrkt::alloc_result(#name, [&] { return (expr); }); \
  }

extern "C" {

TR_UNARY_OP(tr_sum, t->value.sum())
TR_UNARY_OP(tr_mean, t->value.mean())
TR_UNARY_OP(tr_max, t->value.max())
TR_UNARY_OP(tr_min, t->value.min())
TR_UNARY_OP(tr_argmax_all, t->value.argmax())

tr_tensor* tr_argmax(const tr_tensor* t, int64_t dim, int keepdim) {
  if (!t) {
    return torchrkt::null_arg("tr_argmax");
  }
  return torchrkt::alloc_result(
      "tr_argmax", [&] { return t->value.argmax(dim, keepdim != 0); });
}

tr_tensor* tr_softmax(const tr_tensor* t, int64_t dim) {
  if (!t) {
    return torchrkt::null_arg("tr_softmax");
  }
  return torchrkt::alloc_result("tr_softmax",
                                [&] { return t->value.softmax(dim); });
}

tr_tensor* tr_log_softmax(const tr_tensor* t, int64_t dim) {
  if (!t) {
    return torchrkt::null_arg("tr_log_softmax");
  }
  return torchrkt::alloc_result("tr_log_softmax",
                                [&] { return t->value.log_softmax(dim); });
}

}  // extern "C"

#undef TR_UNARY_OP
