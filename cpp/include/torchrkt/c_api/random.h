#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate a CPU float32 tensor of i.i.d. standard-normal N(0,1) draws with the
 * given shape, seeded by the current global RNG (see tr_manual_seed). `dims`
 * points at `ndim` sizes (ndim==0 yields a scalar). Returns a new tr_tensor the
 * caller must release with tr_tensor_free, or NULL on error (see
 * tr_last_error). The return-handle-or-NULL shape lets the Racket FFI manage it
 * with the standard allocator/deallocator finalizer pair. */
tr_tensor* tr_randn(const int64_t* dims, int64_t ndim);

/* Like tr_randn but i.i.d. uniform on [0, 1) (torch.rand). */
tr_tensor* tr_rand(const int64_t* dims, int64_t ndim);

/* Fill `t` in place with i.i.d. uniform draws on [low, high), consuming the
 * global RNG exactly like torch.Tensor.uniform_ — the primitive behind
 * nn.init.kaiming_uniform_, so seeded init parity with PyTorch holds. */
int tr_tensor_uniform_(tr_tensor* t, double low, double high);

#ifdef __cplusplus
}
#endif
