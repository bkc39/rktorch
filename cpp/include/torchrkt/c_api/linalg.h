#pragma once

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Linear algebra. Each returns a new tr_tensor handle (NULL on error).
 * tr_matmul follows torch.matmul's broadcasting rules; tr_mm is strictly
 * 2D x 2D, tr_mv is 2D x 1D, tr_dot is 1D x 1D. */

tr_tensor* tr_matmul(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_mm(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_mv(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_dot(const tr_tensor* a, const tr_tensor* b);

#ifdef __cplusplus
}
#endif
