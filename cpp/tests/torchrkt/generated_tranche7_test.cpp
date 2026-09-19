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

std::vector<int64_t> indices_of(const tr_tensor* t) {
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

void expect_error_from(const char* who) {
  const char* message = tr_last_error();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(std::strstr(message, who), nullptr) << message;
}

TEST(GeneratedTranche7, TopkWritesValuesAndIndicesThroughItsOutPointers) {
  const Handle input = make({1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 6.0F}, {2, 3});
  tr_tensor* values = nullptr;
  tr_tensor* indices = nullptr;
  ASSERT_EQ(tr_gen_topk(input.t, 2, -1, true, true, &values, &indices), 0)
      << tr_last_error();
  const Handle owned_values(values);
  const Handle owned_indices(indices);
  EXPECT_EQ(shape_of(values), (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(data_of(values), (std::vector<float>{5.0F, 3.0F, 6.0F, 4.0F}));
  EXPECT_EQ(indices_of(indices), (std::vector<int64_t>{1, 2, 2, 0}));
}

TEST(GeneratedTranche7, SortWritesValuesAndIndicesThroughItsOutPointers) {
  const Handle input = make({3.0F, 1.0F, 2.0F}, {3});
  tr_tensor* values = nullptr;
  tr_tensor* indices = nullptr;
  ASSERT_EQ(tr_gen_sort(input.t, 0, true, &values, &indices), 0)
      << tr_last_error();
  const Handle owned_values(values);
  const Handle owned_indices(indices);
  EXPECT_EQ(data_of(values), (std::vector<float>{3.0F, 2.0F, 1.0F}));
  EXPECT_EQ(indices_of(indices), (std::vector<int64_t>{0, 2, 1}));
}

TEST(GeneratedTranche7, AFailedCallLeavesEveryOutPointerNull) {
  const Handle input = make({1.0F, 2.0F, 3.0F}, {3});
  tr_tensor* values = input.t;
  tr_tensor* indices = input.t;
  EXPECT_EQ(tr_gen_topk(input.t, 4, 0, true, true, &values, &indices), 1);
  expect_error_from("tr_gen_topk");
  EXPECT_EQ(values, nullptr);
  EXPECT_EQ(indices, nullptr);
}

TEST(GeneratedTranche7, ARefusedCallNullsEveryOutSlotItCanReach) {
  const Handle input = make({1.0F, 2.0F, 3.0F}, {3});
  tr_tensor* values = input.t;
  tr_tensor* indices = input.t;
  EXPECT_EQ(tr_gen_topk(nullptr, 1, 0, true, true, &values, &indices), 1);
  expect_error_from("tr_gen_topk");
  EXPECT_EQ(values, nullptr);
  EXPECT_EQ(indices, nullptr);
  indices = input.t;
  EXPECT_EQ(tr_gen_topk(input.t, 1, 0, true, true, nullptr, &indices), 1);
  expect_error_from("tr_gen_topk");
  EXPECT_EQ(indices, nullptr);
  values = input.t;
  EXPECT_EQ(tr_gen_sort(input.t, 0, false, &values, nullptr), 1);
  expect_error_from("tr_gen_sort");
  EXPECT_EQ(values, nullptr);
}

}  // namespace
