#include "torchrkt/c_api/image.h"

#include <stb_image.h>
#include <torch/torch.h>

#include <climits>
#include <memory>
#include <stdexcept>
#include <string>

#include "torchrkt/detail/jpeg.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

struct StbiFree {
  void operator()(stbi_uc* pixels) const noexcept {
    stbi_image_free(pixels);
  }
};

using PixelPtr = std::unique_ptr<stbi_uc, StbiFree>;

[[noreturn]] void decode_failure() {
  const char* why = stbi_failure_reason();
  throw std::runtime_error(std::string("cannot decode image: ") +
                           (why != nullptr ? why : "unknown reason"));
}

torch::Tensor decode_stb(const uint8_t* data, int len, int32_t channels) {
  int width = 0;
  int height = 0;
  int stored = 0;
  if (stbi_info_from_memory(data, len, &width, &height, &stored) == 0) {
    decode_failure();
  }
  torchrkt::detail::check_pixels(width, height);
  const PixelPtr pixels(
      stbi_load_from_memory(data, len, &width, &height, &stored, channels));
  if (!pixels) {
    decode_failure();
  }
  const int kept = channels == 0 ? stored : channels;
  return torch::from_blob(pixels.get(), {height, width, kept}, torch::kUInt8)
      .clone();
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
    const torch::Tensor hwc =
        torchrkt::detail::is_jpeg(data, static_cast<size_t>(len))
            ? torchrkt::detail::decode_jpeg(data, static_cast<size_t>(len),
                                            channels)
            : decode_stb(data, static_cast<int>(len), channels);
    return hwc.permute({2, 0, 1});
  });
}

}  // extern "C"
