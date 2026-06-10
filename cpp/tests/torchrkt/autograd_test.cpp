#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "torchrkt/c_api.h"

// The autograd boundary: d(sum(x*x))/dx == 2x, grad-mode scoping, the
// storage-sharing contract of tr_tensor_grad, and the in-place SGD
// primitives.

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

std::vector<float> data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, 0, nullptr, &numel), 2) << tr_last_error();
  std::vector<float> out(numel);
  EXPECT_EQ(tr_tensor_copy_data(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

TEST(TorchrktAutograd, GradOfSumOfSquaresIsTwoX) {
  Handle x = make({1.0F, 2.0F, 3.0F}, {3});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();

  int requires_grad = 0;
  ASSERT_EQ(tr_tensor_requires_grad(x.t, &requires_grad), 0) << tr_last_error();
  EXPECT_EQ(requires_grad, 1);

  const Handle sq(tr_mul(x.t, x.t));
  Handle y(tr_sum(sq.t));
  ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();

  const Handle g(tr_tensor_grad(x.t));
  EXPECT_EQ(data_of(g.t), (std::vector<float>{2.0F, 4.0F, 6.0F}));
}

TEST(TorchrktAutograd, GradBeforeBackwardErrors) {
  Handle x = make({1.0F}, {1});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  EXPECT_EQ(tr_tensor_grad(x.t), nullptr);
}

TEST(TorchrktAutograd, HasGradPredicate) {
  Handle x = make({1.0F, 2.0F}, {2});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();

  int has = -1;
  ASSERT_EQ(tr_tensor_has_grad(x.t, &has), 0) << tr_last_error();
  EXPECT_EQ(has, 0);

  const Handle sq(tr_mul(x.t, x.t));
  Handle y(tr_sum(sq.t));
  ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();

  ASSERT_EQ(tr_tensor_has_grad(x.t, &has), 0) << tr_last_error();
  EXPECT_EQ(has, 1);

  EXPECT_EQ(tr_tensor_has_grad(nullptr, &has), 1);
}

TEST(TorchrktAutograd, InplaceScalarMultiply) {
  Handle x = make({1.0F, -2.0F}, {2});
  ASSERT_EQ(tr_tensor_mul_(x.t, 2.5), 0) << tr_last_error();
  EXPECT_EQ(data_of(x.t), (std::vector<float>{2.5F, -5.0F}));

  // NULL surfaces as a status code, never an abort.
  EXPECT_EQ(tr_tensor_mul_(nullptr, 2.0), 1);
}

TEST(TorchrktAutograd, GradModeScopesRecording) {
  int enabled = -1;
  ASSERT_EQ(tr_is_grad_enabled(&enabled), 0) << tr_last_error();
  EXPECT_EQ(enabled, 1);

  Handle x = make({1.0F}, {1});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();

  ASSERT_EQ(tr_set_grad_enabled(0), 0) << tr_last_error();
  const Handle y(tr_mul(x.t, x.t));
  ASSERT_EQ(tr_set_grad_enabled(1), 0) << tr_last_error();

  // y was computed outside grad mode: backward through it must fail.
  Handle s(tr_sum(y.t));
  EXPECT_EQ(tr_tensor_backward(s.t), 1);
}

TEST(TorchrktAutograd, GradSharesStorageAndInplaceOpsWork) {
  Handle x = make({1.0F, 2.0F}, {2});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  {
    const Handle sq(tr_mul(x.t, x.t));
    Handle y(tr_sum(sq.t));
    ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
  }

  // Zeroing through one grad handle is visible through a fresh one.
  {
    const Handle g(tr_tensor_grad(x.t));
    ASSERT_EQ(tr_tensor_zero_(g.t), 0) << tr_last_error();
  }
  const Handle g2(tr_tensor_grad(x.t));
  EXPECT_EQ(data_of(g2.t), (std::vector<float>{0.0F, 0.0F}));

  // The SGD primitive p -= alpha*other under grad mode off.
  ASSERT_EQ(tr_set_grad_enabled(0), 0) << tr_last_error();
  const Handle step = make({1.0F, 1.0F}, {2});
  ASSERT_EQ(tr_tensor_sub_(x.t, step.t, 0.5), 0) << tr_last_error();
  ASSERT_EQ(tr_set_grad_enabled(1), 0) << tr_last_error();
  EXPECT_EQ(data_of(x.t), (std::vector<float>{0.5F, 1.5F}));
}

}  // namespace
