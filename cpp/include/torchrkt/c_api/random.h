#pragma once

#include <stdint.h>

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_randn(const int64_t* dims, int64_t ndim);

tr_tensor* tr_rand(const int64_t* dims, int64_t ndim);

/* torch.randn / torch.rand with the device and dtype chosen at construction
 * (TR_DEVICE_KEEP / TR_DTYPE_KEEP: the process default device / float32);
 * both draw from the chosen device's generator. */
tr_tensor* tr_randn_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                       int64_t index, tr_dtype dtype);

tr_tensor* tr_rand_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                      int64_t index, tr_dtype dtype);

int tr_tensor_uniform_(tr_tensor* t, double low, double high);

#ifdef __cplusplus
}
#endif
