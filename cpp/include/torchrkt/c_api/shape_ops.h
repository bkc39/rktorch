#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_reshape(const tr_tensor* t, const int64_t* dims, int64_t ndim);

tr_tensor* tr_view(const tr_tensor* t, const int64_t* dims, int64_t ndim);

tr_tensor* tr_transpose(const tr_tensor* t, int64_t dim0, int64_t dim1);

tr_tensor* tr_permute(const tr_tensor* t, const int64_t* dims, int64_t ndim);

tr_tensor* tr_squeeze(const tr_tensor* t);

tr_tensor* tr_squeeze_dim(const tr_tensor* t, int64_t dim);

tr_tensor* tr_unsqueeze(const tr_tensor* t, int64_t dim);

tr_tensor* tr_cat(const tr_tensor* const* tensors, int64_t n, int64_t dim);

tr_tensor* tr_stack(const tr_tensor* const* tensors, int64_t n, int64_t dim);

#ifdef __cplusplus
}
#endif
