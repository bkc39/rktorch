#pragma once

#include <string>

namespace torchrkt {

// Machine-readable classification of the last error, paired with the
// message. The int values are the C contract (tr_last_error_kind in
// c_api/global.h) — keep the two docs in sync.
// NOLINTNEXTLINE(performance-enum-size) -- int IS the C ABI contract here
enum class error_kind : int { generic = 0, oom = 1 };

// Thread-local last-error string, surfaced to Racket via tr_last_error().
extern thread_local std::string g_last_error;
extern thread_local error_kind g_last_error_kind;

void set_error(const std::string& message);
void set_error(const std::string& message, error_kind kind);

// The exhaustion-safe recorder: never allocates on its own failure path, so
// it is callable inside a noexcept boundary when the rich message build has
// itself failed. Message is best-effort; the given (pre-classified) kind is
// always recorded.
void set_error_fallback(const char* who, error_kind kind) noexcept;

inline void set_error_oom(const char* who) noexcept {
  set_error_fallback(who, error_kind::oom);
}

std::string last_error();
error_kind last_error_kind();

}  // namespace torchrkt
