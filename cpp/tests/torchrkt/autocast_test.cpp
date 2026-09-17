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

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

tr_dtype dtype_of(const tr_tensor* t) {
  tr_dtype dt = TR_DTYPE_KEEP;
  EXPECT_EQ(tr_tensor_dtype(t, &dt), 0) << tr_last_error();
  return dt;
}

std::vector<float> data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, 0, nullptr, &numel), 2) << tr_last_error();
  std::vector<float> out(numel);
  EXPECT_EQ(tr_tensor_copy_data(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

TEST(TorchrktAutocast, OffByDefaultWithTheCpuDefaultDtype) {
  int enabled = -1;
  ASSERT_EQ(tr_is_autocast_enabled(TR_DEVICE_CPU, &enabled), 0)
      << tr_last_error();
  EXPECT_EQ(enabled, 0);
  tr_dtype dt = TR_DTYPE_KEEP;
  ASSERT_EQ(tr_autocast_dtype(TR_DEVICE_CPU, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_BFLOAT16);
}

TEST(TorchrktAutocast, CpuBfloat16CastsMatmulUntilDisabled) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const Handle b = make({0.5F, 0.0F, 0.0F, 0.5F}, {2, 2});
  ASSERT_EQ(tr_set_autocast_enabled(TR_DEVICE_CPU, TR_DTYPE_BFLOAT16, 1), 0)
      << tr_last_error();
  int enabled = 0;
  ASSERT_EQ(tr_is_autocast_enabled(TR_DEVICE_CPU, &enabled), 0);
  EXPECT_EQ(enabled, 1);
  {
    const Handle cast(tr_matmul(a.t, b.t));
    EXPECT_EQ(dtype_of(cast.t), TR_DTYPE_BFLOAT16);
    EXPECT_EQ(data_of(cast.t), (std::vector<float>{0.5F, 1.0F, 1.5F, 2.0F}));
  }
  ASSERT_EQ(tr_set_autocast_enabled(TR_DEVICE_CPU, TR_DTYPE_KEEP, 0), 0)
      << tr_last_error();
  ASSERT_EQ(tr_is_autocast_enabled(TR_DEVICE_CPU, &enabled), 0);
  EXPECT_EQ(enabled, 0);
  const Handle plain(tr_matmul(a.t, b.t));
  EXPECT_EQ(dtype_of(plain.t), TR_DTYPE_FLOAT32);
  // the inputs were never touched
  EXPECT_EQ(dtype_of(a.t), TR_DTYPE_FLOAT32);
}

TEST(TorchrktAutocast, RejectsNonHalfDtypesAndBadArguments) {
  EXPECT_EQ(tr_set_autocast_enabled(TR_DEVICE_CPU, TR_DTYPE_FLOAT32, 1), 1);
  EXPECT_NE(std::strstr(tr_last_error(), "float16 or bfloat16"), nullptr)
      << tr_last_error();
  int enabled = -1;
  ASSERT_EQ(tr_is_autocast_enabled(TR_DEVICE_CPU, &enabled), 0);
  EXPECT_EQ(enabled, 0) << "a refused enable leaves autocast off";
  EXPECT_EQ(tr_set_autocast_enabled(TR_DEVICE_KEEP, TR_DTYPE_BFLOAT16, 1), 1);
  EXPECT_EQ(tr_is_autocast_enabled(TR_DEVICE_CPU, nullptr), 1);
  EXPECT_EQ(tr_autocast_dtype(TR_DEVICE_CPU, nullptr), 1);
}

}  // namespace
