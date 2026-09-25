#include "torchrkt/c_api/image.h"

#include <stb_image.h>
#include <torch/torch.h>

#include <climits>
#include <memory>
#include <stdexcept>
#include <string>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

struct StbiFree {
  void operator()(stbi_uc* pixels) const noexcept {
    stbi_image_free(pixels);
  }
};

using PixelPtr = std::unique_ptr<stbi_uc, StbiFree>;

// Pillow's DecompressionBombError threshold, twice its warning at
// 89,478,485 pixels: refused from the header, before a pixel is decoded
constexpr int64_t kMaxPixels = 178956970;

[[noreturn]] void decode_failure() {
  const char* why = stbi_failure_reason();
  throw std::runtime_error(std::string("cannot decode image: ") +
                           (why != nullptr ? why : "unknown reason"));
}

void check_size(const uint8_t* data, int len) {
  int width = 0;
  int height = 0;
  int stored = 0;
  if (stbi_info_from_memory(data, len, &width, &height, &stored) == 0) {
    decode_failure();
  }
  const int64_t pixels = static_cast<int64_t>(width) * height;
  if (pixels > kMaxPixels) {
    throw std::invalid_argument(
        "image of " + std::to_string(width) + " x " + std::to_string(height) +
        " pixels is over the " + std::to_string(kMaxPixels) + "-pixel limit");
  }
}

}  // namespace

extern "C" {

tr_tensor* tr_image_decode(const uint8_t* data, int64_t len, int32_t channels) {
  if (data == nullptr) {
    return torchrkt::null_arg("tr_image_decode");
  }
  return torchrkt::alloc_result("tr_image_decode", [&] {
    if (len < 1 || len > INT_MAX) {
      throw std::invalid_argument("encoded length " + std::to_string(len) +
                                  " is outside 1 .. INT_MAX");
    }
    if (channels < 0 || channels > 4) {
      throw std::invalid_argument(
          "channels must be 0 (as stored) through 4, got " +
          std::to_string(channels));
    }
    check_size(data, static_cast<int>(len));
    int width = 0;
    int height = 0;
    int stored = 0;
    const PixelPtr pixels(stbi_load_from_memory(
        data, static_cast<int>(len), &width, &height, &stored, channels));
    if (!pixels) {
      decode_failure();
    }
    const int kept = channels == 0 ? stored : channels;
    // clone, not contiguous: a one-channel permute is already contiguous and
    // would come back as a view of the buffer freed on return
    return torch::from_blob(pixels.get(), {height, width, kept}, torch::kUInt8)
        .permute({2, 0, 1})
        .clone(at::MemoryFormat::Contiguous);
  });
}

}  // extern "C"
