#pragma once

#include <c10/util/Exception.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <tuple>
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
  // Allocator refusals thrown as plain c10::Error, matched by message shape;
  // each phrase has one emitting site in libtorch 2.8-2.12 (re-audit on bump).
  const std::string_view what(e.what());
  if (what.find("DefaultCPUAllocator") != std::string_view::npos ||
      what.find("MPS backend out of memory") != std::string_view::npos ||
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

template <typename Handle, typename Fn>
Handle* alloc_handle(const char* who, Fn&& fn) noexcept {
  try {
    return new Handle{std::forward<Fn>(fn)()};
  } catch (const std::exception& e) {
    record_failure(who, e);
    return nullptr;
  } catch (...) {
    record_unknown_failure(who);
    return nullptr;
  }
}

template <typename Fn>
tr_tensor* alloc_result(const char* who, Fn&& fn) noexcept {
  return alloc_handle<tr_tensor>(who, std::forward<Fn>(fn));
}

// Every handle is built before any out pointer is written, so a throw midway
// frees the ones already made and the caller sees only NULLs.
template <std::size_t N, typename Fn>
int alloc_results(const char* who, tr_tensor** const (&outs)[N],
                  Fn&& fn) noexcept {
  for (tr_tensor** out : outs) {
    *out = nullptr;
  }
  try {
    auto results = std::forward<Fn>(fn)();
    static_assert(std::tuple_size_v<decltype(results)> == N,
                  "one out pointer per Tensor return");
    auto handles = std::apply(
        [](auto&... value) {
          return std::array<std::unique_ptr<tr_tensor>, N>{
              std::unique_ptr<tr_tensor>(new tr_tensor{std::move(value)})...};
        },
        results);
    for (std::size_t i = 0; i < N; ++i) {
      *outs[i] = handles[i].release();
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

// Size-then-fill in bytes. The size is metadata, so a probe with no
// capacity answers without moving the tensor to the host; only a call
// with room materializes it.
inline int copy_tensor_bytes(const char* who, const torch::Tensor& t,
                             uint64_t capacity, uint8_t* out,
                             uint64_t* out_nbytes) noexcept {
  *out_nbytes = 0;
  try {
    const auto nbytes = static_cast<uint64_t>(t.numel()) *
                        static_cast<uint64_t>(t.element_size());
    *out_nbytes = nbytes;
    if (capacity < nbytes) {
      return 2;
    }
    if (out && nbytes > 0) {
      const torch::Tensor c = t.to(torch::kCPU).contiguous();
      std::memcpy(out, c.const_data_ptr(), nbytes);
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

// A refused call still owes the caller NULL in every out slot it can reach.
template <std::size_t N>
int null_arg_outputs(const char* who, tr_tensor** const (&outs)[N]) {
  for (tr_tensor** out : outs) {
    if (out) {
      *out = nullptr;
    }
  }
  return null_arg_status(who);
}

}  // namespace torchrkt
