#pragma once

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int tr_tensor_requires_grad_(tr_tensor* t, int requires_grad);

int tr_tensor_requires_grad(const tr_tensor* t, int* out);

int tr_tensor_has_grad(const tr_tensor* t, int* out);

int tr_tensor_backward(tr_tensor* t);

/* Aliases the live .grad (mutation reaches the optimizer); errors if no
 * gradient has accumulated. */
tr_tensor* tr_tensor_grad(const tr_tensor* t);

tr_tensor* tr_tensor_detach(const tr_tensor* t);

int tr_set_grad_enabled(int enabled);

int tr_is_grad_enabled(int* out);

int tr_tensor_sub_(tr_tensor* t, const tr_tensor* other, double alpha);

int tr_tensor_zero_(tr_tensor* t);

int tr_tensor_mul_(tr_tensor* t, double value);

#ifdef __cplusplus
}
#endif
