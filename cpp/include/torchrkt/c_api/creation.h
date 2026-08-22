#pragma once

#include <stdint.h>

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor constructors. All allocate CPU float32 tensors and follow tr_randn's
 * contract: return a new tr_tensor the caller releases with tr_tensor_free,
 * or NULL on error (see tr_last_error). `dims` points at `ndim` sizes;
 * ndim==0 yields a scalar. */

tr_tensor* tr_zeros(const int64_t* dims, int64_t ndim);

tr_tensor* tr_ones(const int64_t* dims, int64_t ndim);

/* Every element set to `value`. */
tr_tensor* tr_full(const int64_t* dims, int64_t ndim, double value);

/* Values in [start, end) with the given stride, like torch.arange (always
 * float32 here, where PyTorch would infer int64 from integer arguments). */
tr_tensor* tr_arange(double start, double end, double step);

/* n x m identity (ones on the main diagonal). */
tr_tensor* tr_eye(int64_t n, int64_t m);

/* Build a tensor from `numel` row-major float32 values reshaped to `dims`.
 * numel must equal the product of dims; the data is copied. */
tr_tensor* tr_from_data(const float* data, uint64_t numel, const int64_t* dims,
                        int64_t ndim);

tr_tensor* tr_from_data_on_device(const float* data, uint64_t numel,
                                  const int64_t* dims, int64_t ndim,
                                  tr_device_type device_type,
                                  int64_t device_index);

tr_tensor* tr_from_data_i64(const int64_t* data, uint64_t numel,
                            const int64_t* dims, int64_t ndim);

tr_tensor* tr_from_data_i64_on_device(const int64_t* data, uint64_t numel,
                                      const int64_t* dims, int64_t ndim,
                                      tr_device_type device_type,
                                      int64_t device_index);

#ifdef __cplusplus
}
#endif
