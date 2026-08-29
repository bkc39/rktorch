#include "torchrkt/c_api/spectral.h"

#include <torch/torch.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "torchrkt/detail/device.hpp"
#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

extern "C" {

tr_tensor* tr_hann_window(int64_t window_length, bool periodic) {
  return torchrkt::alloc_result("tr_hann_window", [&] {
    if (window_length < 0) {
      throw std::invalid_argument("window length must be nonnegative, got " +
                                  std::to_string(window_length));
    }
    return torch::hann_window(window_length, periodic,
                              torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torchrkt::current_default_device()));
  });
}

// The complex STFT viewed as real: (..., freq, frames, 2), last dim
// re/im — the caller composes magnitude or power without a complex
// dtype crossing the C boundary. hop/win of -1 select torch's
// defaults (n_fft/4 and n_fft).
tr_tensor* tr_stft(const tr_tensor* self, int64_t n_fft, int64_t hop_length,
                   int64_t win_length, const tr_tensor* window, bool center,
                   bool normalized) {
  if (self == nullptr) {
    return torchrkt::null_arg("tr_stft");
  }
  return torchrkt::alloc_result("tr_stft", [&] {
    if (n_fft <= 0) {
      throw std::invalid_argument("n_fft must be positive, got " +
                                  std::to_string(n_fft));
    }
    if (hop_length < -1 || hop_length == 0) {
      throw std::invalid_argument("hop length must be positive or -1, got " +
                                  std::to_string(hop_length));
    }
    if (win_length < -1 || win_length == 0) {
      throw std::invalid_argument("win length must be positive or -1, got " +
                                  std::to_string(win_length));
    }
    const std::optional<int64_t> hop =
        hop_length == -1 ? std::nullopt : std::optional<int64_t>(hop_length);
    const std::optional<int64_t> win =
        win_length == -1 ? std::nullopt : std::optional<int64_t>(win_length);
    // materializing the rectangular default silences libtorch's
    // spectral-leakage warning without changing the result; it matches
    // self's options because torch::stft requires window dtype == input
    const std::optional<torch::Tensor> w =
        window == nullptr ? std::optional<torch::Tensor>(torch::ones(
                                win.value_or(n_fft), self->value.options()))
                          : std::optional<torch::Tensor>(window->value);
    const torch::Tensor c =
        torch::stft(self->value, n_fft, hop, win, w, center, "reflect",
                    normalized, /*onesided=*/true, /*return_complex=*/true);
    return torch::view_as_real(c).contiguous();
  });
}

}  // extern "C"
