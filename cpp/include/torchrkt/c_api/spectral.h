#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TR_DEVICE_KEEP and TR_DTYPE_KEEP select the process default device and
 * float32, so the window is built where it will be used rather than moved
 * there afterwards. */
tr_tensor* tr_hann_window(int64_t window_length, bool periodic,
                          tr_device_type type, int64_t index, tr_dtype dtype);
tr_tensor* tr_stft(const tr_tensor* self, int64_t n_fft, int64_t hop_length,
                   int64_t win_length, const tr_tensor* window, bool center,
                   bool normalized);

#ifdef __cplusplus
}
#endif
