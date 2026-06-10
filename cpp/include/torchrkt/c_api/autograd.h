#pragma once

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Autograd. Boolean parameters/results are 0/1 ints. */

/* Flip requires_grad in place (torch.Tensor.requires_grad_). Only valid on
 * leaf floating-point tensors, like in PyTorch. */
int tr_tensor_requires_grad_(tr_tensor* t, int requires_grad);

int tr_tensor_requires_grad(const tr_tensor* t, int* out);

/* Backpropagate from a scalar tensor (torch.Tensor.backward). */
int tr_tensor_backward(tr_tensor* t);

/* Handle on the accumulated gradient. The returned tensor SHARES storage
 * with the live .grad (it is the same ATen tensor), so in-place ops on it —
 * e.g. tr_tensor_zero_ — affect the gradient the optimizer reads next.
 * Errors if no gradient has been accumulated yet. */
tr_tensor* tr_tensor_grad(const tr_tensor* t);

/* A new handle detached from the autograd graph (torch.Tensor.detach). */
tr_tensor* tr_tensor_detach(const tr_tensor* t);

/* Thread-local grad mode (torch.set_grad_enabled). The Racket FFI calls in
 * on one OS thread, so save/restore from Racket behaves like a dynamic
 * extent. */
int tr_set_grad_enabled(int enabled);

int tr_is_grad_enabled(int* out);

/* In-place ops backing the pure-Racket SGD step (run them under grad mode
 * disabled, as torch.optim does). tr_tensor_sub_ is t -= alpha * other. */
int tr_tensor_sub_(tr_tensor* t, const tr_tensor* other, double alpha);

int tr_tensor_zero_(tr_tensor* t);

int tr_tensor_mul_(tr_tensor* t, double value);

#ifdef __cplusplus
}
#endif
