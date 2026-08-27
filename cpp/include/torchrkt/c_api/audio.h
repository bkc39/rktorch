#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int tr_audio_info(const char* path, int64_t* frames, int32_t* rate,
                  int32_t* channels);
tr_tensor* tr_audio_load(const char* path, int64_t frame_offset,
                         int64_t num_frames, int32_t* rate);
int tr_audio_save(const char* path, const tr_tensor* samples, int32_t rate);

#ifdef __cplusplus
}
#endif
