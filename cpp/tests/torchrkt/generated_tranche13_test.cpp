#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
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

std::vector<uint8_t> bytes_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data_u8(t, 0, nullptr, &numel), 2)
      << tr_last_error();
  std::vector<uint8_t> out(numel);
  EXPECT_EQ(tr_tensor_copy_data_u8(t, numel, out.data(), &numel), 0)
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

Handle make_u8(const std::vector<uint8_t>& values,
               const std::vector<int64_t>& dims) {
  return Handle(tr_from_data_u8(values.data(), values.size(), dims.data(),
                                static_cast<int64_t>(dims.size())));
}

void expect_near(const std::vector<float>& got, const std::vector<float>& want,
                 float tol) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
  }
}

void expect_refused(tr_tensor* got, const char* name) {
  EXPECT_EQ(got, nullptr);
  EXPECT_NE(std::strstr(tr_last_error(), name), nullptr) << tr_last_error();
}

const std::vector<int64_t> kHalf{1, 2};

TEST(GeneratedTranche13, BilinearAveragesTheTwoInputsAroundEachCentre) {
  const Handle row = make({0.0F, 1.0F, 2.0F, 3.0F}, {1, 1, 1, 4});
  const Handle plain(tr_gen_upsample_bilinear2d(row.t, kHalf.data(), 2, false,
                                                0.0, false, 0.0, false));
  EXPECT_EQ(shape_of(plain.t), (std::vector<int64_t>{1, 1, 1, 2}));
  expect_near(data_of(plain.t), {0.5F, 2.5F}, 1e-6F);
  const Handle wide(tr_gen__upsample_bilinear2d_aa(
      row.t, kHalf.data(), 2, false, 0.0, false, 0.0, false));
  expect_near(data_of(wide.t), {5.0F / 7.0F, 16.0F / 7.0F}, 1e-6F);
}

TEST(GeneratedTranche13, BilinearResamplesBytesAsBytes) {
  const Handle row = make_u8({10, 11, 20, 23}, {1, 1, 1, 4});
  for (const bool antialias : {false, true}) {
    const Handle out(
        antialias ? tr_gen__upsample_bilinear2d_aa(
                        row.t, kHalf.data(), 2, false, 0.0, false, 0.0, false)
                  : tr_gen_upsample_bilinear2d(row.t, kHalf.data(), 2, false,
                                               0.0, false, 0.0, false));
    tr_dtype dtype{};
    ASSERT_EQ(tr_tensor_dtype(out.t, &dtype), 0) << tr_last_error();
    EXPECT_EQ(dtype, TR_DTYPE_UINT8);
    const std::vector<uint8_t> got = bytes_of(out.t);
    ASSERT_EQ(got.size(), 2U);
    const std::vector<int> near =
        antialias ? std::vector<int>{12, 20} : std::vector<int>{10, 22};
    EXPECT_LE(std::abs(got[0] - near[0]), 1) << int{got[0]};
    EXPECT_LE(std::abs(got[1] - near[1]), 1) << int{got[1]};
  }
  const Handle flat = make_u8(std::vector<uint8_t>(5 * 7, 200), {1, 1, 5, 7});
  const std::vector<int64_t> odd{3, 2};
  const Handle shrunk(tr_gen__upsample_bilinear2d_aa(
      flat.t, odd.data(), 2, false, 0.0, false, 0.0, false));
  EXPECT_EQ(bytes_of(shrunk.t), std::vector<uint8_t>(6, 200));
}

TEST(GeneratedTranche13, RoundTakesHalvesToEven) {
  const Handle x = make({0.5F, 1.5F, 2.5F, -0.5F, -2.5F, 2.4F}, {6});
  const Handle rounded(tr_gen_round(x.t));
  EXPECT_EQ(data_of(rounded.t),
            (std::vector<float>{0.0F, 2.0F, 2.0F, -0.0F, -2.0F, 2.0F}));
}

TEST(GeneratedTranche13, RefusesAMissingTensorOrANegativeLength) {
  const Handle row = make({0.0F, 1.0F, 2.0F, 3.0F}, {1, 1, 1, 4});
  expect_refused(tr_gen_upsample_bilinear2d(nullptr, kHalf.data(), 2, false,
                                            0.0, false, 0.0, false),
                 "tr_gen_upsample_bilinear2d");
  expect_refused(tr_gen_upsample_bilinear2d(row.t, kHalf.data(), -1, false, 0.0,
                                            false, 0.0, false),
                 "tr_gen_upsample_bilinear2d");
  expect_refused(tr_gen__upsample_bilinear2d_aa(nullptr, kHalf.data(), 2, false,
                                                0.0, false, 0.0, false),
                 "tr_gen__upsample_bilinear2d_aa");
  expect_refused(tr_gen_round(nullptr), "tr_gen_round");
}

}  // namespace
