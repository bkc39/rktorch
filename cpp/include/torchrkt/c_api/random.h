#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_randn(const int64_t* dims, int64_t ndim);

tr_tensor* tr_rand(const int64_t* dims, int64_t ndim);

int tr_tensor_uniform_(tr_tensor* t, double low, double high);

#ifdef __cplusplus
}
#endif
