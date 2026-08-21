#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Device kinds reachable from Racket. Values are part of the C ABI, and so is
 * the enum's width: the Racket FFI marshals it as a C int, so a narrower base
 * type would be an ABI break (C can't specify one before C23). */
/* NOLINTNEXTLINE(performance-enum-size) */
typedef enum tr_device_type {
  TR_DEVICE_CPU = 0,
  TR_DEVICE_CUDA = 1
} tr_device_type;

/* 1 if a CUDA device is present and usable, 0 otherwise. Never returns a status
 * error (a CPU-only libtorch links this and reports 0); but a 0 from a CUDA
 * driver/init failure (rare) also records a message in tr_last_error, so a
 * caller that wants to distinguish "no CUDA" from "CUDA broke" can check it. */
int tr_cuda_is_available(void);

/* Number of visible CUDA devices, 0 when CUDA is unavailable. Same
 * tr_last_error behaviour as tr_cuda_is_available on a driver/init failure. */
int tr_cuda_device_count(void);

/* The device the new-tensor constructors (tr_zeros/tr_randn/...) place results
 * on. Defaults to CPU. Requesting a CUDA device validates availability and the
 * ordinal. 0: success, 1: error (see tr_last_error). */
int tr_set_default_device(tr_device_type type, int64_t index);

/* Read the current default device. 0: success, 1: error. */
int tr_get_default_device(tr_device_type* out_type, int64_t* out_index);

/* Copy a tensor onto the given device (like torch.Tensor.to(device)). Returns
 * a new handle, NULL on error. `index` is the CUDA ordinal, ignored for CPU. */
tr_tensor* tr_tensor_to_device(const tr_tensor* t, tr_device_type type,
                               int64_t index);

/* Report the device a tensor currently lives on. 0: success, 1: error. */
/* Per-device CUDA caching-allocator gauges, in bytes: currently
 * allocated, currently reserved (cached), and peak allocated. 0 with the
 * outputs filled; 1 (see tr_last_error) when CUDA support is not
 * compiled into this build, CUDA is unavailable, or the ordinal is out
 * of range. Complements the Racket-side handle ledger: the ledger is
 * what rktorch holds, these are what the allocator holds. */
int tr_cuda_memory_stats(int64_t device_index, int64_t* out_allocated,
                         int64_t* out_reserved, int64_t* out_peak_allocated);

/* Release the caching allocator's unused cached blocks back to the
 * driver. No-op success when CUDA is unavailable or not compiled in, so
 * it is safe to call unconditionally (the OOM retry does). */
int tr_cuda_empty_cache(void);

int tr_tensor_device(const tr_tensor* t, tr_device_type* out_type,
                     int64_t* out_index);

#ifdef __cplusplus
}
#endif
