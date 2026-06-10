#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "torchrkt/c_api.h"

// Goldens for representative ops per family (creation, shape, elementwise,
// reduce, linalg, marshalling). Exhaustive value parity with PyTorch lives in
// the Racket python-cross-test; these pin the C contract.

namespace {

// RAII so a failed EXPECT can't leak handles past the test body.
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

std::vector<int64_t> shape_of(const tr_tensor* t) {
  int64_t ndim = 0;
  EXPECT_EQ(tr_tensor_ndim(t, &ndim), 0) << tr_last_error();
  std::vector<int64_t> dims(static_cast<size_t>(ndim));
  int64_t got = 0;
  EXPECT_EQ(tr_tensor_shape(t, ndim, dims.data(), &got), 0) << tr_last_error();
  return dims;
}

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

TEST(TorchrktOps, FromDataRoundTrips) {
  const std::vector<float> values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const Handle t = make(values, {2, 3});
  EXPECT_EQ(shape_of(t.t), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(data_of(t.t), values);
}

TEST(TorchrktOps, FromDataRejectsNumelMismatch) {
  const std::vector<float> values = {1.0F, 2.0F, 3.0F};
  const std::vector<int64_t> dims = {2, 2};
  EXPECT_EQ(tr_from_data(values.data(), values.size(), dims.data(), 2),
            nullptr);
}

TEST(TorchrktOps, FromDataRejectsOverflowingShape) {
  // A dim product that wraps uint64 must not bypass the numel check.
  const std::vector<float> values = {1.0F, 2.0F};
  const std::vector<int64_t> dims = {2, 9223372036854775807LL, 2};
  EXPECT_EQ(tr_from_data(values.data(), values.size(), dims.data(), 3),
            nullptr);
  const std::vector<int64_t> negative = {-1, 2};
  EXPECT_EQ(tr_from_data(values.data(), values.size(), negative.data(), 2),
            nullptr);
}

TEST(TorchrktOps, CreationGoldens) {
  const std::vector<int64_t> dims = {2, 2};
  const Handle z(tr_zeros(dims.data(), 2));
  EXPECT_EQ(data_of(z.t), (std::vector<float>{0, 0, 0, 0}));

  const Handle o(tr_ones(dims.data(), 2));
  EXPECT_EQ(data_of(o.t), (std::vector<float>{1, 1, 1, 1}));

  const Handle f(tr_full(dims.data(), 2, 7.5));
  EXPECT_EQ(data_of(f.t), (std::vector<float>{7.5F, 7.5F, 7.5F, 7.5F}));

  const Handle a(tr_arange(0.0, 3.0, 1.0));
  EXPECT_EQ(data_of(a.t), (std::vector<float>{0, 1, 2}));

  const Handle e(tr_eye(2, 2));
  EXPECT_EQ(data_of(e.t), (std::vector<float>{1, 0, 0, 1}));
}

TEST(TorchrktOps, ShapeOps) {
  const Handle t = make({1, 2, 3, 4, 5, 6}, {2, 3});

  const std::vector<int64_t> flat = {6};
  const Handle r(tr_reshape(t.t, flat.data(), 1));
  EXPECT_EQ(shape_of(r.t), flat);

  const std::vector<int64_t> three_two = {3, 2};
  const Handle v(tr_view(t.t, three_two.data(), 2));
  EXPECT_EQ(shape_of(v.t), three_two);
  EXPECT_EQ(data_of(v.t), (std::vector<float>{1, 2, 3, 4, 5, 6}));

  const std::vector<int64_t> perm = {1, 0};
  const Handle p(tr_permute(t.t, perm.data(), 2));
  EXPECT_EQ(shape_of(p.t), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(data_of(p.t), (std::vector<float>{1, 4, 2, 5, 3, 6}));

  const Handle tr(tr_transpose(t.t, 0, 1));
  EXPECT_EQ(shape_of(tr.t), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(data_of(tr.t), (std::vector<float>{1, 4, 2, 5, 3, 6}));

  const Handle u(tr_unsqueeze(t.t, 0));
  EXPECT_EQ(shape_of(u.t), (std::vector<int64_t>{1, 2, 3}));
  const Handle s(tr_squeeze(u.t));
  EXPECT_EQ(shape_of(s.t), (std::vector<int64_t>{2, 3}));
  const Handle sd(tr_squeeze_dim(u.t, 0));
  EXPECT_EQ(shape_of(sd.t), (std::vector<int64_t>{2, 3}));

  const tr_tensor* pair[] = {t.t, t.t};
  const Handle c(tr_cat(pair, 2, 0));
  EXPECT_EQ(shape_of(c.t), (std::vector<int64_t>{4, 3}));
  const Handle st(tr_stack(pair, 2, 0));
  EXPECT_EQ(shape_of(st.t), (std::vector<int64_t>{2, 2, 3}));
}

TEST(TorchrktOps, Elementwise) {
  const Handle a = make({1, -2, 3, -4}, {2, 2});
  const Handle b = make({10, 20, 30, 40}, {2, 2});

  const Handle sum(tr_add(a.t, b.t));
  EXPECT_EQ(data_of(sum.t), (std::vector<float>{11, 18, 33, 36}));

  const Handle scaled(tr_mul_scalar(a.t, 2.0));
  EXPECT_EQ(data_of(scaled.t), (std::vector<float>{2, -4, 6, -8}));

  const Handle rect(tr_relu(a.t));
  EXPECT_EQ(data_of(rect.t), (std::vector<float>{1, 0, 3, 0}));

  const Handle squared(tr_pow_scalar(a.t, 2.0));
  EXPECT_EQ(data_of(squared.t), (std::vector<float>{1, 4, 9, 16}));
}

TEST(TorchrktOps, ReduceAndItem) {
  const Handle t = make({1, 2, 3, 4}, {2, 2});

  const Handle s(tr_sum(t.t));
  double item = 0.0;
  ASSERT_EQ(tr_tensor_item(s.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 10.0);

  const Handle m(tr_mean(t.t));
  ASSERT_EQ(tr_tensor_item(m.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 2.5);

  const Handle mx(tr_max(t.t));
  ASSERT_EQ(tr_tensor_item(mx.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 4.0);

  const Handle mn(tr_min(t.t));
  ASSERT_EQ(tr_tensor_item(mn.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 1.0);

  const Handle am(tr_argmax_all(t.t));
  ASSERT_EQ(tr_tensor_item(am.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 3.0);

  const Handle sm(tr_softmax(t.t, 1));
  const std::vector<float> probs = data_of(sm.t);
  EXPECT_NEAR(probs[0] + probs[1], 1.0F, 1e-6F);
  EXPECT_NEAR(probs[2] + probs[3], 1.0F, 1e-6F);

  // item on a multi-element tensor must error, not crash.
  EXPECT_EQ(tr_tensor_item(t.t, &item), 1);
}

TEST(TorchrktOps, Linalg) {
  const Handle a = make({1, 2, 3, 4}, {2, 2});
  const Handle b = make({5, 6, 7, 8}, {2, 2});
  const Handle v = make({1, 1}, {2});

  const Handle mm(tr_matmul(a.t, b.t));
  EXPECT_EQ(data_of(mm.t), (std::vector<float>{19, 22, 43, 50}));

  const Handle mv(tr_mv(a.t, v.t));
  EXPECT_EQ(data_of(mv.t), (std::vector<float>{3, 7}));

  const Handle d(tr_dot(v.t, v.t));
  double item = 0.0;
  ASSERT_EQ(tr_tensor_item(d.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 2.0);

  // Shape mismatch surfaces as NULL + message, not an abort.
  const Handle bad = make({1, 2, 3}, {3});
  EXPECT_EQ(tr_mv(a.t, bad.t), nullptr);
}

TEST(TorchrktOps, ToDtypeAndUniform) {
  const Handle t = make({1.5F, 2.5F}, {2});
  const Handle i(tr_tensor_to_dtype(t.t, TR_DTYPE_INT64));
  // Reading an INT64 tensor through data_of is safe by contract:
  // tr_tensor_copy_data converts to CPU float32 contiguous before copying
  // (see c_api/tensor.h), so this is a conversion, not a reinterpret.
  EXPECT_EQ(data_of(i.t), (std::vector<float>{1, 2}));

  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  const std::vector<int64_t> dims = {32};
  const Handle u(tr_rand(dims.data(), 1));
  for (const float x : data_of(u.t)) {
    EXPECT_GE(x, 0.0F);
    EXPECT_LT(x, 1.0F);
  }

  Handle w(tr_zeros(dims.data(), 1));
  ASSERT_EQ(tr_tensor_uniform_(w.t, -2.0, -1.0), 0) << tr_last_error();
  for (const float x : data_of(w.t)) {
    EXPECT_GE(x, -2.0F);
    EXPECT_LT(x, -1.0F);
  }
}

}  // namespace
