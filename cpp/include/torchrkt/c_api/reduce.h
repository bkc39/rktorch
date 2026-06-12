#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reductions. Each returns a new tr_tensor handle (NULL on error). The
 * whole-tensor forms reduce to a 0-dim scalar tensor (read it with
 * tr_tensor_item). */

tr_tensor* tr_sum(const tr_tensor* t);
tr_tensor* tr_mean(const tr_tensor* t);
tr_tensor* tr_max(const tr_tensor* t);
tr_tensor* tr_min(const tr_tensor* t);

/* Index of the maximum over the flattened tensor (0-dim int64 tensor). */
tr_tensor* tr_argmax_all(const tr_tensor* t);

/* Index of the maximum along `dim` (int64 tensor). keepdim: 0 or 1. */
tr_tensor* tr_argmax(const tr_tensor* t, int64_t dim, int keepdim);

tr_tensor* tr_softmax(const tr_tensor* t, int64_t dim);
tr_tensor* tr_log_softmax(const tr_tensor* t, int64_t dim);

#ifdef __cplusplus
}
#endif
