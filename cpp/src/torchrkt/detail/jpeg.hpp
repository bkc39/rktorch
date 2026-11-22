#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace torchrkt::detail {

constexpr int64_t kMaxPixels = 178956970;

inline void check_pixels(int64_t width, int64_t height) {
  if (width * height > kMaxPixels) {
    throw std::invalid_argument(
        "image of " + std::to_string(width) + " x " + std::to_string(height) +
        " pixels is over the " + std::to_string(kMaxPixels) + "-pixel limit");
  }
}

inline bool is_jpeg(const uint8_t* data, size_t len) {
  return len >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff;
}

torch::Tensor decode_jpeg(const uint8_t* data, size_t len, int32_t channels);

}  // namespace torchrkt::detail
