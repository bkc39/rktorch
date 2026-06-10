#pragma once

#include <exception>
#include <string>
#include <utility>

#include "torchrkt/c_api/tensor.h"
#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

// Boundary helpers shared by every op translation unit. Each op body reduces
// to a null-argument guard plus one of these wrappers, so the
// exception-to-status contract lives in exactly one place.

namespace torchrkt {

// Run `fn` (returning a torch::Tensor) and wrap the result in a fresh heap
// handle; on any exception, stash the message and return NULL.
template <typename Fn>
tr_tensor* alloc_result(const char* who, Fn&& fn) noexcept {
  try {
    return new tr_tensor{std::forward<Fn>(fn)()};
  } catch (const std::exception& e) {
    set_error(std::string(who) + ": " + e.what());
    return nullptr;
  } catch (...) {
    set_error(std::string(who) + ": unknown exception");
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
    set_error(std::string(who) + ": " + e.what());
    return 1;
  } catch (...) {
    set_error(std::string(who) + ": unknown exception");
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
