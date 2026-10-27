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

/* Grad mode is per calling thread (c10::GradMode), not process-wide. */
int tr_set_grad_enabled(int enabled);

int tr_is_grad_enabled(int* out);

int tr_tensor_sub_(tr_tensor* t, const tr_tensor* other, double alpha);

int tr_tensor_zero_(tr_tensor* t);

int tr_tensor_mul_(tr_tensor* t, double value);

int tr_tensors_reset_grad(tr_tensor* const* tensors, int64_t n);

/* One handle per tensor in grads[i] (as tr_tensor_grad), NULL where tensors[i]
 * has no gradient, with that gradient's tr_tensor_nbytes and tr_tensor_device
 * in nbytes[i], device_type[i] and device_index[i]. On failure the status is
 * 1 and every grads[i] is NULL. */
int tr_tensors_grad(tr_tensor* const* tensors, int64_t n, tr_tensor** grads,
                    int64_t* nbytes, int* device_type, int64_t* device_index);

/* tr_tensor_free over a list; NULL entries are skipped. */
void tr_tensors_free(tr_tensor* const* tensors, int64_t n);

#ifdef __cplusplus
}
#endif
