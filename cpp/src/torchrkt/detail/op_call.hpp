#pragma once

#include <c10/util/Exception.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <string_view>
#include <utility>

#include "torchrkt/c_api/tensor.h"
#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace torchrkt {

inline error_kind classify(const std::exception& e) noexcept {
  if (dynamic_cast<const c10::OutOfMemoryError*>(&e) != nullptr) {
    return error_kind::oom;
  }
  if (dynamic_cast<const std::bad_alloc*>(&e) != nullptr) {
    return error_kind::oom;
  }
  if (std::string_view(e.what()).find("DefaultCPUAllocator") !=
      std::string_view::npos) {
    return error_kind::oom;
  }
  // MPS allocation refusals are plain c10::Errors (TORCH_CHECK), so only
  // these two message shapes identify them as OOM.
  const std::string_view what(e.what());
  if (what.find("MPS backend out of memory") != std::string_view::npos ||
      what.find("Invalid buffer size:") != std::string_view::npos) {
    return error_kind::oom;
  }
  return error_kind::generic;
}

inline void record_failure(const char* who, const std::exception& e) noexcept {
  const error_kind kind = classify(e);
  try {
    set_error(std::string(who) + ": " + e.what(), kind);
  } catch (...) {
    set_error_fallback(who, kind);
  }
}

inline void record_unknown_failure(const char* who) noexcept {
  try {
    set_error(std::string(who) + ": unknown exception");
  } catch (...) {
    set_error_fallback(who, error_kind::generic);
  }
}

template <typename Fn>
tr_tensor* alloc_result(const char* who, Fn&& fn) noexcept {
  try {
    return new tr_tensor{std::forward<Fn>(fn)()};
  } catch (const std::exception& e) {
    record_failure(who, e);
    return nullptr;
  } catch (...) {
    record_unknown_failure(who);
    return nullptr;
  }
}

template <typename Fn>
int status_call(const char* who, Fn&& fn) noexcept {
  try {
    std::forward<Fn>(fn)();
    return 0;
  } catch (const std::exception& e) {
    record_failure(who, e);
    return 1;
  } catch (...) {
    record_unknown_failure(who);
    return 1;
  }
}

template <typename Scalar, typename Fn>
int copy_data_call(const char* who, uint64_t capacity, Scalar* out,
                   uint64_t* out_numel, Fn&& fn) noexcept {
  *out_numel = 0;
  try {
    const torch::Tensor c = std::forward<Fn>(fn)();
    const auto numel = static_cast<uint64_t>(c.numel());
    *out_numel = numel;
    if (capacity < numel) {
      return 2;
    }
    if (out && numel > 0) {
      std::memcpy(out, c.template data_ptr<Scalar>(), numel * sizeof(Scalar));
    }
    return 0;
  } catch (const std::exception& e) {
    record_failure(who, e);
    return 1;
  } catch (...) {
    record_unknown_failure(who);
    return 1;
  }
}

inline tr_tensor* null_arg(const char* who) {
  set_error(std::string(who) + ": null argument");
  return nullptr;
}

inline int null_arg_status(const char* who) {
  set_error(std::string(who) + ": null argument");
  return 1;
}

}  // namespace torchrkt
