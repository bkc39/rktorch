#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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

void expect_error_from(const char* who) {
  const char* message = tr_last_error();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(std::strstr(message, who), nullptr) << message;
}

void expect_near(const std::vector<float>& got, const std::vector<float>& want,
                 float tol) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
  }
}

TEST(GeneratedTranche5,
     RepeatInterleaveAlongBothSpatialDimsIsNearestUpsampling) {
  const Handle input = make({1.0F, 2.0F, 3.0F, 4.0F}, {1, 1, 2, 2});
  const Handle rows(
      tr_gen_repeat_interleave_self_int(input.t, 2, 2, true, 0, false));
  EXPECT_EQ(shape_of(rows.t), (std::vector<int64_t>{1, 1, 4, 2}));
  expect_near(data_of(rows.t), {1.0F, 2.0F, 1.0F, 2.0F, 3.0F, 4.0F, 3.0F, 4.0F},
              0.0F);
  const Handle both(
      tr_gen_repeat_interleave_self_int(rows.t, 2, 3, true, 0, false));
  EXPECT_EQ(shape_of(both.t), (std::vector<int64_t>{1, 1, 4, 4}));
  expect_near(data_of(both.t),
              {1.0F, 1.0F, 2.0F, 2.0F, 1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F,
               4.0F, 3.0F, 3.0F, 4.0F, 4.0F},
              0.0F);
  const Handle sized(
      tr_gen_repeat_interleave_self_int(input.t, 2, 2, true, 4, true));
  EXPECT_EQ(shape_of(sized.t), (std::vector<int64_t>{1, 1, 4, 2}));
  const Handle flat(
      tr_gen_repeat_interleave_self_int(input.t, 2, 0, false, 0, false));
  EXPECT_EQ(shape_of(flat.t), (std::vector<int64_t>{8}));
  expect_near(data_of(flat.t), {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F},
              0.0F);
  EXPECT_EQ(tr_gen_repeat_interleave_self_int(nullptr, 2, 2, true, 0, false),
            nullptr);
  expect_error_from("tr_gen_repeat_interleave_self_int");
}

}  // namespace
