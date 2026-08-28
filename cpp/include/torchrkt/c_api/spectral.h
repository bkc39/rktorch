#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_hann_window(int64_t window_length, bool periodic);
tr_tensor* tr_stft(const tr_tensor* self, int64_t n_fft, int64_t hop_length,
                   int64_t win_length, const tr_tensor* window, bool center,
                   bool normalized);

#ifdef __cplusplus
}
#endif
