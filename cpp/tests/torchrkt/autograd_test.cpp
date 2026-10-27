#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "torchrkt/c_api.h"

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

  {
    const Handle g(tr_tensor_grad(x.t));
    ASSERT_EQ(tr_tensor_zero_(g.t), 0) << tr_last_error();
  }
  const Handle g2(tr_tensor_grad(x.t));
  EXPECT_EQ(data_of(g2.t), (std::vector<float>{0.0F, 0.0F}));

  ASSERT_EQ(tr_set_grad_enabled(0), 0) << tr_last_error();
  const Handle step = make({1.0F, 1.0F}, {2});
  ASSERT_EQ(tr_tensor_sub_(x.t, step.t, 0.5), 0) << tr_last_error();
  ASSERT_EQ(tr_set_grad_enabled(1), 0) << tr_last_error();
  EXPECT_EQ(data_of(x.t), (std::vector<float>{0.5F, 1.5F}));
}

TEST(TorchrktAutograd, ResetGradDropsEveryGradientInOneCall) {
  Handle x = make({1.0F, 2.0F}, {2});
  Handle w = make({3.0F}, {1});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  ASSERT_EQ(tr_tensor_requires_grad_(w.t, 1), 0) << tr_last_error();
  auto backward = [&] {
    const Handle sq(tr_mul(x.t, x.t));
    const Handle wx(tr_mul(sq.t, w.t));
    Handle y(tr_sum(wx.t));
    ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
  };
  backward();
  const Handle before(tr_tensor_grad(x.t));

  std::vector<tr_tensor*> both{x.t, w.t};
  ASSERT_EQ(tr_tensors_reset_grad(both.data(), 2), 0) << tr_last_error();
  int has = -1;
  for (tr_tensor* t : both) {
    ASSERT_EQ(tr_tensor_has_grad(t, &has), 0) << tr_last_error();
    EXPECT_EQ(has, 0);
  }

  backward();
  const Handle after(tr_tensor_grad(x.t));
  EXPECT_EQ(data_of(after.t), (std::vector<float>{6.0F, 12.0F}));
  EXPECT_EQ(data_of(before.t), (std::vector<float>{6.0F, 12.0F}));

  ASSERT_EQ(tr_tensors_reset_grad(both.data(), 2), 0) << tr_last_error();
  ASSERT_EQ(tr_tensors_reset_grad(both.data(), 2), 0) << tr_last_error();
  EXPECT_EQ(tr_tensors_reset_grad(nullptr, 0), 0) << tr_last_error();
}

TEST(TorchrktAutograd, ResetGradRejectsNullArguments) {
  Handle x = make({1.0F}, {1});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  const Handle y(tr_sum(x.t));
  ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
  std::vector<tr_tensor*> with_null{x.t, nullptr};
  const std::string null_argument = "tr_tensors_reset_grad: null argument";

  EXPECT_EQ(tr_tensors_reset_grad(nullptr, 1), 1);
  EXPECT_EQ(tr_last_error(), null_argument);
  EXPECT_EQ(tr_tensors_reset_grad(with_null.data(), 2), 1);
  EXPECT_EQ(tr_last_error(), null_argument);
  int has = -1;
  ASSERT_EQ(tr_tensor_has_grad(x.t, &has), 0) << tr_last_error();
  EXPECT_EQ(has, 1) << "a refused list reset an element before its NULL";

  EXPECT_EQ(tr_tensors_reset_grad(with_null.data(), -1), 1);
  EXPECT_EQ(std::string(tr_last_error()),
            "tr_tensors_reset_grad: negative length");
}

TEST(TorchrktAutograd, TensorsGradMatchesTheOneAtATimeQueries) {
  Handle x = make({1.0F, 2.0F}, {2});
  Handle w = make({3.0F}, {1});
  Handle idle = make({5.0F}, {1});
  for (tr_tensor* t : {x.t, w.t, idle.t}) {
    ASSERT_EQ(tr_tensor_requires_grad_(t, 1), 0) << tr_last_error();
  }
  {
    const Handle sq(tr_mul(x.t, x.t));
    const Handle wx(tr_mul(sq.t, w.t));
    Handle y(tr_sum(wx.t));
    ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
  }
  std::vector<tr_tensor*> ts{x.t, idle.t, w.t};
  std::vector<tr_tensor*> grads(3, nullptr);
  std::vector<int64_t> nbytes(3, -1);
  std::vector<int> types(3, -1);
  std::vector<int64_t> indices(3, -1);
  ASSERT_EQ(tr_tensors_grad(ts.data(), 3, grads.data(), nbytes.data(),
                            types.data(), indices.data()),
            0)
      << tr_last_error();
  EXPECT_EQ(grads[1], nullptr);
  for (size_t i : {size_t{0}, size_t{2}}) {
    ASSERT_NE(grads[i], nullptr);
    const Handle one(tr_tensor_grad(ts[i]));
    EXPECT_EQ(data_of(grads[i]), data_of(one.t));
    int64_t want_nbytes = -1;
    ASSERT_EQ(tr_tensor_nbytes(one.t, &want_nbytes), 0) << tr_last_error();
    EXPECT_EQ(nbytes[i], want_nbytes);
    tr_device_type type = TR_DEVICE_KEEP;
    int64_t index = -1;
    ASSERT_EQ(tr_tensor_device(one.t, &type, &index), 0) << tr_last_error();
    EXPECT_EQ(types[i], static_cast<int>(type));
    EXPECT_EQ(indices[i], index);
  }
  ASSERT_EQ(tr_tensor_zero_(grads[0]), 0) << tr_last_error();
  const Handle live(tr_tensor_grad(x.t));
  EXPECT_EQ(data_of(live.t), (std::vector<float>{0.0F, 0.0F}))
      << "a fetched handle aliases the live gradient";
  tr_tensors_free(grads.data(), 3);
  tr_tensors_free(nullptr, 2);
}

TEST(TorchrktAutograd, TensorsGradRefusesANullListWithEveryOutputNull) {
  Handle x = make({1.0F}, {1});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  const Handle y(tr_sum(x.t));
  ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
  tr_tensor* stale = x.t;
  std::vector<tr_tensor*> grads{stale, stale};
  std::vector<int64_t> nbytes(2);
  std::vector<int> types(2);
  std::vector<int64_t> indices(2);
  const std::string null_argument = "tr_tensors_grad: null argument";

  EXPECT_EQ(tr_tensors_grad(nullptr, 2, grads.data(), nbytes.data(),
                            types.data(), indices.data()),
            1);
  EXPECT_EQ(tr_last_error(), null_argument);
  EXPECT_EQ(grads, (std::vector<tr_tensor*>{nullptr, nullptr}));

  grads = {stale, stale};
  std::vector<tr_tensor*> with_null{x.t, nullptr};
  EXPECT_EQ(tr_tensors_grad(with_null.data(), 2, grads.data(), nbytes.data(),
                            types.data(), indices.data()),
            1);
  EXPECT_EQ(grads, (std::vector<tr_tensor*>{nullptr, nullptr}));

  EXPECT_EQ(tr_tensors_grad(nullptr, 0, nullptr, nullptr, nullptr, nullptr), 0)
      << tr_last_error();
}

}  // namespace
