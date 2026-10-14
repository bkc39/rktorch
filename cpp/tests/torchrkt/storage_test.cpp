#include <gtest/gtest.h>

#include <cstdint>
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

Handle from_values(const std::vector<float>& values,
                   const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

std::uint64_t data_ptr_of(const tr_tensor* t) {
  std::uint64_t out = 0;
  EXPECT_EQ(tr_tensor_data_ptr(t, &out), 0) << tr_last_error();
  return out;
}

std::uint64_t storage_ptr_of(const tr_tensor* t) {
  std::uint64_t out = 0;
  EXPECT_EQ(tr_tensor_storage_ptr(t, &out), 0) << tr_last_error();
  return out;
}

}  // namespace

TEST(TorchrktStorage, ViewsShareTheStorageAndOffsetTheirDataPointer) {
  const Handle base = from_values({1, 2, 3, 4, 5, 6}, {2, 3});
  const std::vector<int64_t> flat_dims = {6};
  const Handle whole(tr_view(base.t, flat_dims.data(), 1));
  const Handle tail(tr_gen_narrow(base.t, 0, 1, 1));

  EXPECT_NE(data_ptr_of(base.t), 0U);
  EXPECT_EQ(data_ptr_of(whole.t), data_ptr_of(base.t));
  EXPECT_EQ(data_ptr_of(tail.t), data_ptr_of(base.t) + 3 * sizeof(float));
  EXPECT_EQ(storage_ptr_of(tail.t), storage_ptr_of(base.t));
  EXPECT_EQ(storage_ptr_of(base.t), data_ptr_of(base.t));
}

TEST(TorchrktStorage, ACopyHasItsOwnStorage) {
  const Handle base = from_values({1, 2, 3}, {3});
  const Handle copy(tr_tensor_to_dtype(base.t, TR_DTYPE_FLOAT64));

  EXPECT_NE(data_ptr_of(copy.t), data_ptr_of(base.t));
  EXPECT_NE(storage_ptr_of(copy.t), storage_ptr_of(base.t));
}

TEST(TorchrktStorage, AnInPlaceMoveRebindsTheStorageOnlyWhenItMoves) {
  const Handle t = from_values({1, 2, 3}, {3});
  const std::uint64_t before = data_ptr_of(t.t);

  ASSERT_EQ(tr_tensor_to_(t.t, TR_DEVICE_KEEP, -1, TR_DTYPE_FLOAT32), 0)
      << tr_last_error();
  EXPECT_EQ(data_ptr_of(t.t), before);

  ASSERT_EQ(tr_tensor_to_(t.t, TR_DEVICE_KEEP, -1, TR_DTYPE_FLOAT64), 0)
      << tr_last_error();
  EXPECT_NE(data_ptr_of(t.t), before);
  EXPECT_EQ(storage_ptr_of(t.t), data_ptr_of(t.t));
}

TEST(TorchrktStorage, NullArgumentsReportAnError) {
  const Handle t = from_values({1}, {1});
  std::uint64_t out = 0;
  EXPECT_EQ(tr_tensor_data_ptr(nullptr, &out), 1);
  EXPECT_EQ(tr_tensor_data_ptr(t.t, nullptr), 1);
  EXPECT_EQ(tr_tensor_storage_ptr(nullptr, &out), 1);
  EXPECT_EQ(tr_tensor_storage_ptr(t.t, nullptr), 1);
}
