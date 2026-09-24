#pragma once

#include <stdint.h>

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_image_decode(const uint8_t* data, int64_t len, int32_t channels);

#ifdef __cplusplus
}
#endif
