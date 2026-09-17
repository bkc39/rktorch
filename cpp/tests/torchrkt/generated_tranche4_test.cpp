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

const std::vector<int64_t> kOne{1, 1};
const std::vector<int64_t> kZero{0, 0};
const std::vector<int64_t> kTwo{2, 2};

tr_tensor* transpose2d(const tr_tensor* input, const tr_tensor* weight,
                       const tr_tensor* bias,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& output_padding) {
  return tr_gen_conv_transpose2d_input(input, weight, bias, stride.data(), 2,
                                       kZero.data(), 2, output_padding.data(),
                                       2, 1, kOne.data(), 2);
}

TEST(GeneratedTranche4, ConvTranspose2dScattersEachPixelOverTheKernel) {
  const Handle input = make({1.0F, 2.0F, 3.0F, 4.0F}, {1, 1, 2, 2});
  const Handle weight = make({1.0F, 1.0F, 1.0F, 1.0F}, {1, 1, 2, 2});
  const Handle out(transpose2d(input.t, weight.t, nullptr, kOne, kZero));
  EXPECT_EQ(shape_of(out.t), (std::vector<int64_t>{1, 1, 3, 3}));
  EXPECT_EQ(data_of(out.t), (std::vector<float>{1.0F, 3.0F, 2.0F, 4.0F, 10.0F,
                                                6.0F, 3.0F, 7.0F, 4.0F}));
  const Handle bias = make({0.5F}, {1});
  const Handle biased(transpose2d(input.t, weight.t, bias.t, kOne, kZero));
  const std::vector<float> base = data_of(out.t);
  const std::vector<float> shifted = data_of(biased.t);
  for (size_t i = 0; i < base.size(); ++i) {
    EXPECT_FLOAT_EQ(shifted[i], base[i] + 0.5F);
  }
  const Handle strided(transpose2d(input.t, weight.t, nullptr, kTwo, kZero));
  EXPECT_EQ(shape_of(strided.t), (std::vector<int64_t>{1, 1, 4, 4}));
  EXPECT_EQ(
      data_of(strided.t),
      (std::vector<float>{1.0F, 1.0F, 2.0F, 2.0F, 1.0F, 1.0F, 2.0F, 2.0F, 3.0F,
                          3.0F, 4.0F, 4.0F, 3.0F, 3.0F, 4.0F, 4.0F}));
  const Handle padded(transpose2d(input.t, weight.t, nullptr, kTwo, kOne));
  EXPECT_EQ(shape_of(padded.t), (std::vector<int64_t>{1, 1, 5, 5}));
  // the weight is laid out (in, out, kH, kW): two in-channels feed three
  const Handle two_in = make(std::vector<float>(8, 1.0F), {1, 2, 2, 2});
  const Handle to_three = make(std::vector<float>(24, 1.0F), {2, 3, 2, 2});
  const Handle wide(transpose2d(two_in.t, to_three.t, nullptr, kOne, kZero));
  EXPECT_EQ(shape_of(wide.t), (std::vector<int64_t>{1, 3, 3, 3}));
  EXPECT_EQ(transpose2d(nullptr, weight.t, nullptr, kOne, kZero), nullptr);
  expect_error_from("tr_gen_conv_transpose2d_input");
}

TEST(GeneratedTranche4, GroupNormNormalisesPerGroupThenAffines) {
  const Handle input = make({1.0F, 3.0F, 5.0F, 7.0F}, {1, 2, 1, 2});
  const Handle two(tr_gen_group_norm(input.t, 2, nullptr, nullptr, 1e-5, true));
  expect_near(data_of(two.t), {-1.0F, 1.0F, -1.0F, 1.0F}, 1e-4F);
  // one group: mean 4, biased variance 5
  const Handle one(tr_gen_group_norm(input.t, 1, nullptr, nullptr, 1e-5, true));
  expect_near(data_of(one.t), {-1.341641F, -0.447214F, 0.447214F, 1.341641F},
              1e-4F);
  const Handle weight = make({2.0F, 2.0F}, {2});
  const Handle bias = make({1.0F, 1.0F}, {2});
  const Handle affine(
      tr_gen_group_norm(input.t, 2, weight.t, bias.t, 1e-5, true));
  const std::vector<float> bare = data_of(two.t);
  const std::vector<float> scaled = data_of(affine.t);
  for (size_t i = 0; i < bare.size(); ++i) {
    EXPECT_NEAR(scaled[i], 2.0F * bare[i] + 1.0F, 1e-4F);
  }
  EXPECT_EQ(tr_gen_group_norm(nullptr, 2, nullptr, nullptr, 1e-5, true),
            nullptr);
  expect_error_from("tr_gen_group_norm");
}

TEST(GeneratedTranche4, SiluIsXTimesSigmoid) {
  const Handle x = make({0.0F, 1.0F, -1.0F, 2.0F}, {4});
  const Handle y(tr_gen_silu(x.t));
  expect_near(data_of(y.t), {0.0F, 0.7310586F, -0.2689414F, 1.7615942F}, 1e-5F);
  EXPECT_EQ(tr_gen_silu(nullptr), nullptr);
  expect_error_from("tr_gen_silu");
}

TEST(GeneratedTranche4, ClampBoundsAreOptionalButNotBothAbsent) {
  const Handle x = make({-2.0F, -0.5F, 0.0F, 0.5F, 2.0F}, {5});
  const Handle lo(tr_gen_clamp(x.t, -1.0, true, 0.0, false));
  EXPECT_EQ(data_of(lo.t),
            (std::vector<float>{-1.0F, -0.5F, 0.0F, 0.5F, 2.0F}));
  const Handle hi(tr_gen_clamp(x.t, 0.0, false, 1.0, true));
  EXPECT_EQ(data_of(hi.t),
            (std::vector<float>{-2.0F, -0.5F, 0.0F, 0.5F, 1.0F}));
  const Handle both(tr_gen_clamp(x.t, -1.0, true, 1.0, true));
  EXPECT_EQ(data_of(both.t),
            (std::vector<float>{-1.0F, -0.5F, 0.0F, 0.5F, 1.0F}));
  // ATen: with min above max every element becomes max
  const Handle crossed(tr_gen_clamp(x.t, 1.0, true, -1.0, true));
  EXPECT_EQ(data_of(crossed.t),
            (std::vector<float>{-1.0F, -1.0F, -1.0F, -1.0F, -1.0F}));
  EXPECT_EQ(tr_gen_clamp(x.t, 0.0, false, 0.0, false), nullptr);
  expect_error_from("tr_gen_clamp");
  EXPECT_EQ(tr_gen_clamp(nullptr, -1.0, true, 1.0, true), nullptr);
  expect_error_from("tr_gen_clamp");
}

}  // namespace
