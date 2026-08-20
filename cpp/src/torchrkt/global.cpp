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
  buf = torchrkt::last_error();
  return buf.c_str();
}

int tr_last_error_kind(void) {
  return static_cast<int>(torchrkt::last_error_kind());
}

int tr_manual_seed(uint64_t seed) {
  try {
    torch::manual_seed(seed);
    return 0;
  } catch (const std::exception& e) {
    torchrkt::set_error(std::string("tr_manual_seed: ") + e.what(),
                        torchrkt::classify(e));
    return 1;
  } catch (...) {
    torchrkt::set_error("tr_manual_seed: unknown exception");
    return 1;
  }
}

}  // extern "C"
