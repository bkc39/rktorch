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

// Every record updates message AND kind together (the one-argument form
// records generic) so the pair can never desync across boundary calls.
void set_error(const std::string& message);
void set_error(const std::string& message, error_kind kind);

// The exhaustion-safe recorder: never allocates on its own failure path,
// so it is callable from a bad_alloc catch inside a noexcept boundary
// (where a throwing message build would be std::terminate). Message is
// best-effort; the OOM kind is always recorded.
void set_error_oom(const char* who) noexcept;

std::string last_error();
error_kind last_error_kind();

}  // namespace torchrkt
