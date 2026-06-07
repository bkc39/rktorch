#pragma once

#include <string>

namespace torchrkt {

// Thread-local last-error string, surfaced to Racket via tr_last_error().
extern thread_local std::string g_last_error;

void set_error(const std::string& message);

std::string last_error();

}  // namespace torchrkt
