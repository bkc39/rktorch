#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <stdexcept>

#include "torchrkt/c_api.h"
#include "torchrkt/detail/tensor_handle.hpp"

// Pins tr_tensor_free's no-catch invariant: a throwing storage release
// reaches std::terminate inside libtorch's implicitly-noexcept frames before
// any catch at this layer (the live guard is Racket-side, raw/memory.rkt).
namespace {

// The catch is the tripwire: if a future libtorch makes the release path
// catchable, the child exits 0 and EXPECT_DEATH flips loudly.
void free_throwing_tensor_then_exit() {
  static float data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  auto* t = new tr_tensor{torch::from_blob(
      data, {4}, [](void*) { throw std::runtime_error("boom"); },
      torch::TensorOptions().dtype(torch::kFloat32))};
  try {
    tr_tensor_free(t);
  } catch (...) {
    std::exit(0);
  }
  std::exit(0);
}

}  // namespace

TEST(TorchrktFinalizer, ThrowingReleaseTerminatesDespiteAnyCatch) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(free_throwing_tensor_then_exit(), "");
}
