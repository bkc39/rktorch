#include "torchrkt/c_api/elementwise.h"

#include <torch/torch.h>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

// Every op here is the same null-guard + alloc_result shape; the macros keep
// one op per stanza without hand-copying the boundary plumbing.

#define TR_BINARY_OP(name, expr)                                  \
  tr_tensor* name(const tr_tensor* a, const tr_tensor* b) {       \
    if (!a || !b) {                                               \
      return torchrkt::null_arg(#name);                           \
    }                                                             \
    return torchrkt::alloc_result(#name, [&] { return (expr); }); \
  }

#define TR_SCALAR_OP(name, expr)                                  \
  tr_tensor* name(const tr_tensor* a, double b) {                 \
    if (!a) {                                                     \
      return torchrkt::null_arg(#name);                           \
    }                                                             \
    return torchrkt::alloc_result(#name, [&] { return (expr); }); \
  }

#define TR_UNARY_OP(name, expr)                                   \
  tr_tensor* name(const tr_tensor* t) {                           \
    if (!t) {                                                     \
      return torchrkt::null_arg(#name);                           \
    }                                                             \
    return torchrkt::alloc_result(#name, [&] { return (expr); }); \
  }

extern "C" {

TR_BINARY_OP(tr_add, a->value.add(b->value))
TR_BINARY_OP(tr_sub, a->value.sub(b->value))
TR_BINARY_OP(tr_mul, a->value.mul(b->value))
TR_BINARY_OP(tr_div, a->value.div(b->value))
TR_BINARY_OP(tr_pow, a->value.pow(b->value))

TR_SCALAR_OP(tr_add_scalar, a->value.add(b))
TR_SCALAR_OP(tr_sub_scalar, a->value.sub(b))
TR_SCALAR_OP(tr_mul_scalar, a->value.mul(b))
TR_SCALAR_OP(tr_div_scalar, a->value.div(b))
TR_SCALAR_OP(tr_pow_scalar, a->value.pow(b))

TR_UNARY_OP(tr_neg, t->value.neg())
TR_UNARY_OP(tr_exp, t->value.exp())
TR_UNARY_OP(tr_log, t->value.log())
TR_UNARY_OP(tr_sqrt, t->value.sqrt())
TR_UNARY_OP(tr_relu, t->value.relu())
TR_UNARY_OP(tr_sigmoid, t->value.sigmoid())
TR_UNARY_OP(tr_tanh, t->value.tanh())
// gelu's kwarg-only `str approximate` arg is outside the codegen IR; the
// one-arg free function (no method variant exists) is the 'none' default.
TR_UNARY_OP(tr_gelu, at::gelu(t->value))

}  // extern "C"

#undef TR_BINARY_OP
#undef TR_SCALAR_OP
#undef TR_UNARY_OP
