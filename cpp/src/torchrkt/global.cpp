#include "torchrkt/c_api/global.h"

#include <ATen/Context.h>
#include <torch/torch.h>
#include <torch/version.h>

#include <array>
#include <cstddef>
#include <exception>
#include <string>

#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/op_call.hpp"

namespace {

template <typename Read>
int read_flag(const char* who, int* out, Read read) {
  if (!out) {
    return torchrkt::null_arg_status(who);
  }
  return torchrkt::status_call(
      who, [&] { *out = static_cast<int>(read(at::globalContext())); });
}

}  // namespace

extern "C" {

const char* tr_version(void) {
  static thread_local std::string buf;
  buf = std::to_string(TORCH_VERSION_MAJOR) + "." +
        std::to_string(TORCH_VERSION_MINOR) + "." +
        std::to_string(TORCH_VERSION_PATCH);
  return buf.c_str();
}

const char* tr_abi_stamp(void) {
  return TORCHRKT_ABI_STAMP;
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

int tr_set_deterministic_algorithms(int enabled, int warn_only) {
  return torchrkt::status_call("tr_set_deterministic_algorithms", [=] {
    at::globalContext().setDeterministicAlgorithms(enabled != 0,
                                                   warn_only != 0);
  });
}

int tr_get_deterministic_algorithms(int* out) {
  return read_flag(
      "tr_get_deterministic_algorithms", out,
      [](const at::Context& c) { return c.deterministicAlgorithms(); });
}

int tr_get_deterministic_algorithms_warn_only(int* out) {
  return read_flag(
      "tr_get_deterministic_algorithms_warn_only", out,
      [](const at::Context& c) { return c.deterministicAlgorithmsWarnOnly(); });
}

int tr_set_cudnn_deterministic(int enabled) {
  return torchrkt::status_call("tr_set_cudnn_deterministic", [=] {
    at::globalContext().setDeterministicCuDNN(enabled != 0);
  });
}

int tr_get_cudnn_deterministic(int* out) {
  return read_flag("tr_get_cudnn_deterministic", out,
                   [](const at::Context& c) { return c.deterministicCuDNN(); });
}

int tr_set_cudnn_benchmark(int enabled) {
  return torchrkt::status_call("tr_set_cudnn_benchmark", [=] {
    at::globalContext().setBenchmarkCuDNN(enabled != 0);
  });
}

int tr_get_cudnn_benchmark(int* out) {
  return read_flag("tr_get_cudnn_benchmark", out,
                   [](const at::Context& c) { return c.benchmarkCuDNN(); });
}

int tr_set_float32_matmul_precision(int precision) {
  return torchrkt::status_call("tr_set_float32_matmul_precision", [=] {
    constexpr std::array<const char*, 3> names = {"highest", "high", "medium"};
    TORCH_CHECK(precision >= 0 && precision < static_cast<int>(names.size()),
                "no float32 matmul precision has code ", precision);
    at::globalContext().setFloat32MatmulPrecision(
        names.at(static_cast<std::size_t>(precision)));
  });
}

int tr_get_float32_matmul_precision(int* out) {
  return read_flag(
      "tr_get_float32_matmul_precision", out,
      [](const at::Context& c) { return c.float32MatmulPrecision(); });
}

int tr_set_cuda_matmul_allow_tf32(int enabled) {
  return torchrkt::status_call("tr_set_cuda_matmul_allow_tf32", [=] {
    at::globalContext().setAllowTF32CuBLAS(enabled != 0);
  });
}

int tr_get_cuda_matmul_allow_tf32(int* out) {
  return read_flag("tr_get_cuda_matmul_allow_tf32", out,
                   [](const at::Context& c) { return c.allowTF32CuBLAS(); });
}

int tr_set_cudnn_allow_tf32(int enabled) {
  return torchrkt::status_call("tr_set_cudnn_allow_tf32", [=] {
    at::globalContext().setAllowTF32CuDNN(enabled != 0);
  });
}

int tr_get_cudnn_allow_tf32(int* out) {
  return read_flag("tr_get_cudnn_allow_tf32", out,
                   [](const at::Context& c) { return c.allowTF32CuDNN(); });
}

}  // extern "C"
