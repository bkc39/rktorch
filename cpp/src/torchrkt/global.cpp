#include "torchrkt/c_api/global.h"

#include <torch/torch.h>
#include <torch/version.h>

#include <exception>
#include <string>

#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/op_call.hpp"

extern "C" {

const char* tr_version(void) {
  static thread_local std::string buf;
  buf = std::to_string(TORCH_VERSION_MAJOR) + "." +
        std::to_string(TORCH_VERSION_MINOR) + "." +
        std::to_string(TORCH_VERSION_PATCH);
  return buf.c_str();
}

const char* tr_last_error(void) {
  static thread_local std::string buf;
  // The copy can allocate mid-exhaustion; the literal fallback cannot.
  try {
    buf = torchrkt::last_error();
  } catch (...) {
    return "torchrkt: message unavailable under memory exhaustion";
  }
  return buf.c_str();
}

int tr_last_error_kind(void) {
  return static_cast<int>(torchrkt::last_error_kind());
}

int tr_manual_seed(uint64_t seed) {
  return torchrkt::status_call("tr_manual_seed", [seed] {
    torch::manual_seed(seed);
    // The C++ torch::manual_seed covers CPU + CUDA only (unlike Python's).
    if (torch::mps::is_available()) {
      torch::mps::manual_seed(seed);
    }
  });
}

}  // extern "C"
