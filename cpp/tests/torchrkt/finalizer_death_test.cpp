#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <stdexcept>

#include "torchrkt/c_api.h"
#include "torchrkt/detail/tensor_handle.hpp"

// White-box death test pinning why tr_tensor_free carries no try/catch
// (#38): a from_blob deleter that throws stands in for the CUDA caching
// allocator failing during storage release. The throw unwinds through
// libtorch's own implicitly-noexcept frames (TensorBase's noexcept
// move-assign, ~TensorImpl/~StorageImpl/~DataPtr) and reaches
// std::terminate before ANY catch at our layer — catch-based versions of
// tr_tensor_free still aborted this child. The live finalizer-safety
// guarantee is Racket-side (raw/syntax.rkt's deallocator wrap). If a
// libtorch upgrade ever makes the release path catchable, this test fails
// — the signal to reconsider a C++-side catch.
namespace {

// The child's whole life: build a handle whose storage release throws,
// free it through the boundary function. (The exit(0) is unreachable
// today; it exists so a future catchable-release libtorch flips this
// test loudly instead of silently.)
void free_throwing_tensor_then_exit() {
  static float data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  auto* t = new tr_tensor{torch::from_blob(
      data, {4}, [](void*) { throw std::runtime_error("boom"); },
      torch::TensorOptions().dtype(torch::kFloat32))};
  tr_tensor_free(t);
  std::exit(0);
}

}  // namespace

TEST(TorchrktFinalizer, ThrowingReleaseTerminatesDespiteAnyCatch) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(free_throwing_tensor_then_exit(), "");
}
