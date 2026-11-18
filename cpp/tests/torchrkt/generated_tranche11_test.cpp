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

std::vector<int64_t> i64_data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data_i64(t, 0, nullptr, &numel), 2)
      << tr_last_error();
  std::vector<int64_t> out(numel);
  EXPECT_EQ(tr_tensor_copy_data_i64(t, numel, out.data(), &numel), 0)
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

Handle make_i64(const std::vector<int64_t>& values,
                const std::vector<int64_t>& dims) {
  return Handle(tr_from_data_i64(values.data(), values.size(), dims.data(),
                                 static_cast<int64_t>(dims.size())));
}

std::vector<int64_t> drawn(tr_generator* g, int64_t from, int64_t to,
                           bool to_has) {
  const Handle t = make_i64(std::vector<int64_t>(64, -1), {64});
  EXPECT_EQ(tr_gen_random__from(t.t, from, to, to_has, g), 0)
      << tr_last_error();
  return i64_data_of(t.t);
}

TEST(GeneratedTranche11, ConstantPadNdPadsTheLastDimsFirst) {
  const Handle x = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const std::vector<int64_t> pad{1, 0, 0, 2};
  const Handle padded(tr_gen_constant_pad_nd(x.t, pad.data(), 4, 0.0));
  EXPECT_EQ(shape_of(padded.t), (std::vector<int64_t>{4, 3}));
  EXPECT_EQ(data_of(padded.t),
            (std::vector<float>{0.0F, 1.0F, 2.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F,
                                0.0F, 0.0F, 0.0F, 0.0F}));
  const std::vector<int64_t> trim{0, -1};
  const Handle trimmed(tr_gen_constant_pad_nd(x.t, trim.data(), 2, 7.0));
  EXPECT_EQ(data_of(trimmed.t), (std::vector<float>{1.0F, 3.0F}));
  EXPECT_EQ(tr_gen_constant_pad_nd(nullptr, pad.data(), 4, 0.0), nullptr);
  EXPECT_NE(std::strstr(tr_last_error(), "tr_gen_constant_pad_nd"), nullptr);
}

TEST(GeneratedTranche11, RandomFromDrawsInItsRangeFromTheGeneratorItIsHanded) {
  tr_generator* first = tr_generator_new(7);
  tr_generator* second = tr_generator_new(7);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int64_t> a = drawn(first, 3, 9, true);
  EXPECT_EQ(a, drawn(second, 3, 9, true));
  EXPECT_NE(a, drawn(first, 3, 9, true));
  bool seen_low = false;
  bool seen_high = false;
  for (const int64_t v : a) {
    EXPECT_GE(v, 3);
    EXPECT_LT(v, 9);
    seen_low = seen_low || v == 3;
    seen_high = seen_high || v == 8;
  }
  EXPECT_TRUE(seen_low && seen_high);
  tr_generator_free(first);
  tr_generator_free(second);
  for (const int64_t v : drawn(nullptr, 0, 2, true)) {
    EXPECT_TRUE(v == 0 || v == 1) << v;
  }
  for (const int64_t v : drawn(nullptr, 5, 0, false)) {
    EXPECT_GE(v, 5);
  }
  EXPECT_EQ(tr_gen_random__from(nullptr, 0, 2, true, nullptr), 1);
  EXPECT_NE(std::strstr(tr_last_error(), "tr_gen_random__from"), nullptr);
}

}  // namespace
