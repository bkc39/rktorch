#include "torchrkt/c_api/linalg.h"

#include <torch/torch.h>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

#define TR_BINARY_OP(name, expr)                                  \
  tr_tensor* name(const tr_tensor* a, const tr_tensor* b) {       \
    if (!a || !b) {                                               \
      return torchrkt::null_arg(#name);                           \
    }                                                             \
    return torchrkt::alloc_result(#name, [&] { return (expr); }); \
  }

extern "C" {

TR_BINARY_OP(tr_matmul, a->value.matmul(b->value))
TR_BINARY_OP(tr_mm, a->value.mm(b->value))
TR_BINARY_OP(tr_mv, a->value.mv(b->value))
TR_BINARY_OP(tr_dot, a->value.dot(b->value))

}  // extern "C"

#undef TR_BINARY_OP
