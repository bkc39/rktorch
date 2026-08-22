#pragma once

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_matmul(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_mm(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_mv(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_dot(const tr_tensor* a, const tr_tensor* b);

#ifdef __cplusplus
}
#endif
