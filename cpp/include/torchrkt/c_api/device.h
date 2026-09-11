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
  TR_DEVICE_CUDA = 1,
  TR_DEVICE_MPS = 2,
  /* "leave the device alone", meaning the tensor's own device for a
   * conversion and the process default device for a constructor. Accepted
   * by tr_tensor_to, tr_tensor_to_, tr_tensor_to_device, tr_zeros_on,
   * tr_ones_on, and tr_full_on; every other entry point rejects it as an
   * unknown device type. */
  TR_DEVICE_KEEP = -1
} tr_device_type;

/* Probes return 0 when the backend is absent AND when init throws; each
 * clears tr_last_error on entry, so a non-empty message is from that call. */
int tr_cuda_is_available(void);

int tr_cuda_device_count(void);

/* Same 0-on-absent / 0-on-throw convention as the CUDA probes. MPS exposes
 * a single device, so there is no device-count counterpart. */
int tr_mps_is_available(void);

/* No-op success when the MPS backend is absent, like tr_cuda_empty_cache. */
int tr_mps_empty_cache(void);

int tr_set_default_device(tr_device_type type, int64_t index);

int tr_get_default_device(tr_device_type* out_type, int64_t* out_index);

tr_tensor* tr_tensor_to_device(const tr_tensor* t, tr_device_type type,
                               int64_t index);

/* torch.Tensor.to(device=, dtype=): one native hop for either or both axes.
 * TR_DEVICE_KEEP / TR_DTYPE_KEEP leave that axis unchanged. When nothing
 * changes the result aliases t's storage (copy=false), like PyTorch. NULL on
 * error. */
tr_tensor* tr_tensor_to(const tr_tensor* t, tr_device_type type, int64_t index,
                        tr_dtype dtype);

/* In-place counterpart, the mechanism behind torch.nn.Module.to: rebinds t's
 * storage through Tensor::set_data under no_grad, so the handle, its
 * requires_grad flag, its leaf-ness and its version counter survive, and an
 * accumulated .grad is rebound the same way. A no-op when nothing changes.
 * The caller's byte accounting for t is stale afterwards (device and nbytes
 * may both differ). 0 success, 1 error. */
int tr_tensor_to_(tr_tensor* t, tr_device_type type, int64_t index,
                  tr_dtype dtype);

int tr_cuda_memory_stats(int64_t device_index, int64_t* out_allocated,
                         int64_t* out_reserved, int64_t* out_peak_allocated);

int tr_cuda_empty_cache(void);

int tr_tensor_device(const tr_tensor* t, tr_device_type* out_type,
                     int64_t* out_index);

#ifdef __cplusplus
}
#endif
