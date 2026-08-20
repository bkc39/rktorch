#include "torchrkt/detail/error.hpp"

#include <string>

namespace torchrkt {

thread_local std::string g_last_error;
thread_local error_kind g_last_error_kind = error_kind::generic;

void set_error(const std::string& message) {
  set_error(message, error_kind::generic);
}

void set_error(const std::string& message, error_kind kind) {
  g_last_error = message;
  g_last_error_kind = kind;
}

std::string last_error() {
  return g_last_error;
}

error_kind last_error_kind() {
  return g_last_error_kind;
}

}  // namespace torchrkt
