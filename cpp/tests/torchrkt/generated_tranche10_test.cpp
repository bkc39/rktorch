#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

std::vector<int64_t> i64_data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data_i64(t, 0, nullptr, &numel), 2)
      << tr_last_error();
  std::vector<int64_t> out(numel);
  EXPECT_EQ(tr_tensor_copy_data_i64(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

tr_dtype dtype_of(const tr_tensor* t) {
  tr_dtype out = TR_DTYPE_KEEP;
  EXPECT_EQ(tr_tensor_dtype(t, &out), 0) << tr_last_error();
  return out;
}

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

Handle make_i64(const std::vector<int64_t>& values,
                const std::vector<int64_t>& dims) {
  return Handle(tr_from_data_i64(values.data(), values.size(), dims.data(),
                                 static_cast<int64_t>(dims.size())));
}

void expect_floats(const std::vector<float>& got,
                   const std::vector<float>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(want[i])) {
      EXPECT_TRUE(std::isnan(got[i])) << "index " << i;
    } else {
      EXPECT_EQ(got[i], want[i]) << "index " << i;
    }
  }
}

void expect_refused(tr_tensor* got, const char* name) {
  EXPECT_EQ(got, nullptr);
  EXPECT_NE(std::strstr(tr_last_error(), name), nullptr) << tr_last_error();
}

const float kNan = std::numeric_limits<float>::quiet_NaN();

TEST(GeneratedTranche10, ExtremaBroadcastARowAgainstAColumn) {
  const Handle column = make({0.0F, 2.0F}, {2, 1});
  const Handle row = make({-1.0F, 1.0F, 3.0F}, {1, 3});
  const Handle larger(tr_gen_maximum(column.t, row.t));
  expect_floats(data_of(larger.t), {0.0F, 1.0F, 3.0F, 2.0F, 2.0F, 3.0F});
  const Handle smaller(tr_gen_minimum(column.t, row.t));
  expect_floats(data_of(smaller.t), {-1.0F, 0.0F, 0.0F, -1.0F, 1.0F, 2.0F});
}

TEST(GeneratedTranche10, ExtremaPropagateANanFromEitherOperand) {
  const Handle a = make({kNan, 1.0F, kNan, 4.0F}, {4});
  const Handle b = make({2.0F, kNan, kNan, 3.0F}, {4});
  const Handle larger(tr_gen_maximum(a.t, b.t));
  expect_floats(data_of(larger.t), {kNan, kNan, kNan, 4.0F});
  const Handle smaller(tr_gen_minimum(a.t, b.t));
  expect_floats(data_of(smaller.t), {kNan, kNan, kNan, 3.0F});
}

TEST(GeneratedTranche10, ExtremaKeepIntegersAndPromoteAMixedPair) {
  const Handle a = make_i64({5, -2, 7}, {3});
  const Handle b = make_i64({3, 4, 7}, {3});
  const Handle larger(tr_gen_maximum(a.t, b.t));
  EXPECT_EQ(dtype_of(larger.t), TR_DTYPE_INT64);
  EXPECT_EQ(i64_data_of(larger.t), (std::vector<int64_t>{5, 4, 7}));
  const Handle f = make({4.5F, -3.0F, 6.5F}, {3});
  const Handle mixed(tr_gen_minimum(a.t, f.t));
  EXPECT_EQ(dtype_of(mixed.t), TR_DTYPE_FLOAT32);
  expect_floats(data_of(mixed.t), {4.5F, -3.0F, 6.5F});
}

TEST(GeneratedTranche10, ExtremaRefuseAMissingTensor) {
  const Handle a = make({1.0F}, {1});
  expect_refused(tr_gen_maximum(nullptr, a.t), "tr_gen_maximum");
  expect_refused(tr_gen_maximum(a.t, nullptr), "tr_gen_maximum");
  expect_refused(tr_gen_minimum(nullptr, a.t), "tr_gen_minimum");
  expect_refused(tr_gen_minimum(a.t, nullptr), "tr_gen_minimum");
}

}  // namespace
