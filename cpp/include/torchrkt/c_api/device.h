#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Values and int width are C ABI: the Racket FFI marshals this as a C int. */
/* NOLINTNEXTLINE(performance-enum-size) */
typedef enum tr_device_type {
  TR_DEVICE_CPU = 0,
  TR_DEVICE_CUDA = 1
} tr_device_type;

int tr_cuda_is_available(void);

int tr_cuda_device_count(void);

int tr_set_default_device(tr_device_type type, int64_t index);

int tr_get_default_device(tr_device_type* out_type, int64_t* out_index);

tr_tensor* tr_tensor_to_device(const tr_tensor* t, tr_device_type type,
                               int64_t index);

int tr_cuda_memory_stats(int64_t device_index, int64_t* out_allocated,
                         int64_t* out_reserved, int64_t* out_peak_allocated);

int tr_cuda_empty_cache(void);

int tr_tensor_device(const tr_tensor* t, tr_device_type* out_type,
                     int64_t* out_index);

#ifdef __cplusplus
}
#endif
