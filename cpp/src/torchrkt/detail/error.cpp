#include "torchrkt/detail/error.hpp"

#include <string>

namespace torchrkt {

thread_local std::string g_last_error;

void set_error(const std::string& message) {
  g_last_error = message;
}

std::string last_error() {
  return g_last_error;
}

}  // namespace torchrkt
