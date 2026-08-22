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

// Boundary helpers shared by every op translation unit: the
// exception-to-status contract lives in exactly one place. Every extern "C"
// function fits one of five shapes: tensor return (alloc_result), integer
// status (status_call), size-then-fill probe (copy_data_call), the
// value-returning CUDA queries in device.cpp (hand-rolled catch: benign value
// + set_error), and the finalizer tr_tensor_free (NO catch — a throw on the
// storage-release path terminates inside libtorch's noexcept frames, so its
// safety guarantee is the Racket-side deallocator wrap in raw/memory.rkt).

namespace torchrkt {

// CUDA/MPS exhaustion throws the typed c10::OutOfMemoryError, but CPU
// allocation failure arrives as a plain c10::Error from alloc_cpu.cpp's
// caffe2-style enforce — hence the message match.
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
  return error_kind::generic;
}

// Classify first, then ATTEMPT the rich message: the concatenation allocates
// and under genuine host exhaustion would throw inside this noexcept frame
// (= std::terminate); the fallback records the classified kind without
// allocating.
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

// Tensor-return shape: a fresh heap handle, or NULL with tr_last_error set.
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

// Integer-status shape: 0 on success, 1 with tr_last_error set.
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

// Size-then-fill probe shape (`fn` yields the contiguous CPU tensor of the
// target scalar type): 0 on success, 2 with *out_numel = required size when
// `capacity` is too small, 1 with tr_last_error set on any exception.
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
