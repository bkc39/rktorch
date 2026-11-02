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

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
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

const std::vector<int64_t> kSize{1, 1, 2, 2};
const std::vector<float> kIdentity{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};

TEST(GeneratedTranche9, AffineGridSpansTheCornersOrTheirCentres) {
  const Handle theta = make(kIdentity, {1, 2, 3});
  const Handle corners(
      tr_gen_affine_grid_generator(theta.t, kSize.data(), 4, true));
  expect_near(data_of(corners.t),
              {-1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F}, 1e-6F);
  const Handle centres(
      tr_gen_affine_grid_generator(theta.t, kSize.data(), 4, false));
  expect_near(data_of(centres.t),
              {-0.5F, -0.5F, 0.5F, -0.5F, -0.5F, 0.5F, 0.5F, 0.5F}, 1e-6F);
}

TEST(GeneratedTranche9, GridSamplerInterpolatesAndReplaysAnIdentityGrid) {
  const Handle input = make({1.0F, 2.0F, 3.0F, 4.0F}, kSize);
  const Handle points = make({0.0F, 0.0F, -1.0F, -1.0F}, {1, 1, 2, 2});
  const Handle bilinear(tr_gen_grid_sampler_2d(input.t, points.t, 0, 0, true));
  expect_near(data_of(bilinear.t), {2.5F, 1.0F}, 1e-6F);
  const Handle near_corner = make({0.4F, 0.4F}, {1, 1, 1, 2});
  const Handle nearest(
      tr_gen_grid_sampler_2d(input.t, near_corner.t, 1, 0, true));
  expect_near(data_of(nearest.t), {4.0F}, 1e-6F);
  const Handle theta = make(kIdentity, {1, 2, 3});
  const Handle grid(
      tr_gen_affine_grid_generator(theta.t, kSize.data(), 4, true));
  const Handle replayed(tr_gen_grid_sampler_2d(input.t, grid.t, 0, 0, true));
  expect_near(data_of(replayed.t), {1.0F, 2.0F, 3.0F, 4.0F}, 1e-6F);
}

TEST(GeneratedTranche9, BicubicOvershootsWhereBilinearWouldNot) {
  const Handle bump = make({0.0F, 1.0F, 0.0F}, {1, 1, 1, 3});
  const std::vector<int64_t> wide{1, 5};
  const Handle resized(tr_gen_upsample_bicubic2d(bump.t, wide.data(), 2, true,
                                                 0.0, false, 0.0, false));
  expect_near(data_of(resized.t), {0.0F, 0.59375F, 1.0F, 0.59375F, 0.0F},
              1e-5F);
}

TEST(GeneratedTranche9, VisionRefusesAMissingTensorOrANegativeLength) {
  const Handle theta = make(kIdentity, {1, 2, 3});
  const Handle input = make({1.0F, 2.0F, 3.0F, 4.0F}, kSize);
  expect_refused(tr_gen_affine_grid_generator(nullptr, kSize.data(), 4, true),
                 "tr_gen_affine_grid_generator");
  expect_refused(tr_gen_affine_grid_generator(theta.t, nullptr, 4, true),
                 "tr_gen_affine_grid_generator");
  expect_refused(tr_gen_affine_grid_generator(theta.t, kSize.data(), -1, true),
                 "tr_gen_affine_grid_generator");
  expect_refused(tr_gen_grid_sampler_2d(nullptr, input.t, 0, 0, true),
                 "tr_gen_grid_sampler_2d");
  expect_refused(tr_gen_grid_sampler_2d(input.t, nullptr, 0, 0, true),
                 "tr_gen_grid_sampler_2d");
  const std::vector<int64_t> wide{2, 2};
  expect_refused(tr_gen_upsample_bicubic2d(nullptr, wide.data(), 2, true, 0.0,
                                           false, 0.0, false),
                 "tr_gen_upsample_bicubic2d");
  expect_refused(tr_gen_upsample_bicubic2d(input.t, nullptr, 2, true, 0.0,
                                           false, 0.0, false),
                 "tr_gen_upsample_bicubic2d");
  expect_refused(tr_gen_upsample_bicubic2d(input.t, wide.data(), -1, true, 0.0,
                                           false, 0.0, false),
                 "tr_gen_upsample_bicubic2d");
}

}  // namespace
