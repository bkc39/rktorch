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

// Boundary helpers shared by every op translation unit. Each op body reduces
// to a null-argument guard plus one of these wrappers, so the
// exception-to-status contract lives in exactly one place.
//
// These cover the three common extern "C" shapes: a tensor return
// (alloc_result), an integer status (status_call), and the size-then-fill
// marshal-out probe (copy_data_call). Two further shapes deviate
// intentionally: value-returning CUDA queries in device.cpp
// (tr_cuda_is_available / tr_cuda_device_count) hand-roll try/catch (catch +
// set_error + return a benign value); and void-returning *finalizer* bindings
// (tr_tensor_free in tensor.cpp) carry NO catch and never set_error — a throw
// on the storage-release path terminates inside libtorch's own noexcept
// frames before any C++ handler (see finalizer_death_test.cpp), so their
// safety guarantee lives in the Racket-side deallocator wrap
// (torch/foreign/raw/memory.rkt). New boundary functions should fit one of
// these five shapes.

namespace torchrkt {

// Classify an exception crossing the boundary for tr_last_error_kind.
// CUDA (and MPS) allocation exhaustion throws the typed subclass
// c10::OutOfMemoryError; CPU allocation failure arrives as a plain
// c10::Error from the caffe2-style enforce in alloc_cpu.cpp, so a
// contained message match covers that shape (see the backend matrix in
// plans/gpu-memory-management.md, leg 1.5).
inline error_kind classify(const std::exception& e) noexcept {
  if (dynamic_cast<const c10::OutOfMemoryError*>(&e) != nullptr) {
    return error_kind::oom;
  }
  // The handle wrapper's own `new tr_tensor` (and any std allocator on the
  // boundary path) fails as std::bad_alloc — an allocation failure too.
  if (dynamic_cast<const std::bad_alloc*>(&e) != nullptr) {
    return error_kind::oom;
  }
  if (std::string_view(e.what()).find("DefaultCPUAllocator") !=
      std::string_view::npos) {
    return error_kind::oom;
  }
  return error_kind::generic;
}

// Record a boundary failure noexcept-safely: classify first, then ATTEMPT
// the rich message — the concatenation itself allocates, and under genuine
// host exhaustion (as opposed to the architectural-VA rejections the tests
// use) it can throw inside a noexcept frame (= std::terminate, the crash
// class this leg exists to eliminate). The fallback records the kind (and
// a best-effort message) without allocating. This covers every failure
// shape, including the CPU allocator's enforce-path c10::Error and
// std::bad_alloc, in one path.
inline void record_failure(const char* who, const std::exception& e) noexcept {
  const error_kind kind = classify(e);
  try {
    set_error(std::string(who) + ": " + e.what(), kind);
  } catch (...) {
    // Preserve the CLASSIFIED kind: a generic failure whose message
    // build coincidentally hit exhaustion still reports generic.
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

// Run `fn` (returning a torch::Tensor) and wrap the result in a fresh heap
// handle; on any exception, stash the message and return NULL.
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

// Run `fn` (returning void) under the integer-status contract: 0 on success,
// 1 with tr_last_error set on any exception.
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

// Run `fn` (returning the contiguous CPU tensor of the target scalar type)
// under the size-then-fill probe contract shared by the tr_tensor_copy_data*
// family: 0 on success, 2 when `capacity` is too small (with *out_numel
// reporting the required size), 1 with tr_last_error set on any exception.
// The conversion copy can be large, so recording is noexcept-safe.
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
