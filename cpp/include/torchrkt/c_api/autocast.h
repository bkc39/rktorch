#pragma once

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Autocast state is per calling thread and per device type (at::autocast),
 * like grad mode. Enabling sets that device type's autocast dtype, which
 * must be TR_DTYPE_FLOAT16 or TR_DTYPE_BFLOAT16, and turns the cast on for
 * every op the thread dispatches until it is disabled; disabling also drops
 * the weight-cast cache, as torch.autocast's exit does. TR_DEVICE_KEEP is
 * rejected. The dtype applies on both paths, so leaving an extent restores
 * the one it entered with: TR_DTYPE_FLOAT16 or TR_DTYPE_BFLOAT16 sets it,
 * TR_DTYPE_KEEP leaves it, and any other dtype is an error; enabling with
 * TR_DTYPE_KEEP is an error too. Changing the dtype drops the cache. */
int tr_set_autocast_enabled(tr_device_type type, tr_dtype dtype, int enabled);

int tr_is_autocast_enabled(tr_device_type type, int* out);

/* The device type's current autocast dtype, set or not (the process default
 * is float16 for CUDA and bfloat16 for the CPU). */
int tr_autocast_dtype(tr_device_type type, tr_dtype* out);

#ifdef __cplusplus
}
#endif
