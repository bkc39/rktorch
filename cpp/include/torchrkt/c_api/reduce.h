#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_sum(const tr_tensor* t);
tr_tensor* tr_mean(const tr_tensor* t);
tr_tensor* tr_max(const tr_tensor* t);
tr_tensor* tr_min(const tr_tensor* t);

tr_tensor* tr_argmax_all(const tr_tensor* t);

tr_tensor* tr_argmax(const tr_tensor* t, int64_t dim, int keepdim);

tr_tensor* tr_softmax(const tr_tensor* t, int64_t dim);
tr_tensor* tr_log_softmax(const tr_tensor* t, int64_t dim);

#ifdef __cplusplus
}
#endif
