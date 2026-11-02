#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include "torchrkt/c_api.h"

// Replaces the global allocation functions for the whole test binary; they
// behave as the defaults until a test arms a one-shot failure on its thread.
namespace {
thread_local bool fail_next_allocation = false;
}  // namespace

void* operator new(std::size_t size) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(size == 0 ? 1 : size)) {
    return p;
  }
  throw std::bad_alloc();
}

void operator delete(void* p) noexcept {
  std::free(p);
}

void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}

namespace {

struct Handle {
  tr_tensor* t;
  explicit Handle(tr_tensor* p) : t(p) {
    EXPECT_NE(t, nullptr) << tr_last_error();
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() {
    tr_tensor_free(t);
  }
};

tr_tensor* armed_grads(tr_tensor* const* ts, int64_t n, tr_tensor** grads,
                       int64_t* nbytes, int* types, int64_t* indices,
                       int* rc) {
  fail_next_allocation = true;
  *rc = tr_tensors_grad(ts, n, grads, nbytes, types, indices);
  fail_next_allocation = false;
  return grads[0];
}

TEST(TorchrktAllocFailure, TensorsGradReportsAHostOomAsAStatus) {
  const std::vector<float> values{1.0F, 2.0F};
  const std::vector<int64_t> dims{2};
  const Handle x(tr_from_data(values.data(), values.size(), dims.data(), 1));
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  {
    const Handle y(tr_sum(x.t));
    ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
  }
  std::vector<tr_tensor*> ts{x.t};
  std::vector<tr_tensor*> grads{nullptr};
  std::vector<int64_t> nbytes{-1};
  std::vector<int> types{-1};
  std::vector<int64_t> indices{-1};
  int rc = -1;

  EXPECT_NO_THROW(armed_grads(ts.data(), 1, grads.data(), nbytes.data(),
                              types.data(), indices.data(), &rc))
      << "a bad_alloc escaped the C boundary";
  EXPECT_EQ(rc, 1);
  EXPECT_EQ(tr_last_error_kind(), 1) << tr_last_error();
  EXPECT_EQ(grads[0], nullptr);
}

}  // namespace
