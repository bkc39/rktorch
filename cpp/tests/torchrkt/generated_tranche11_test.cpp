#include <gtest/gtest.h>

#include <array>
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

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

Handle scalar(float value) {
  return Handle(tr_from_data(&value, 1, nullptr, 0));
}

constexpr int32_t kAtenInt = 3;

Handle int32_scalar(float value) {
  const Handle f = scalar(value);
  return Handle(
      tr_gen_sum_dim_intlist(f.t, nullptr, 0, false, false, kAtenInt));
}

void expect_floats(const std::vector<float>& got,
                   const std::vector<float>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i], want[i]) << "index " << i;
  }
}

void expect_status_refused(int rc, const char* name) {
  EXPECT_EQ(rc, 1);
  EXPECT_NE(std::strstr(tr_last_error(), name), nullptr) << tr_last_error();
}

const float kInf = std::numeric_limits<float>::infinity();
const double kInfD = std::numeric_limits<double>::infinity();

TEST(GeneratedTranche11, ForeachNormAnswersOneNormPerTensor) {
  const Handle a = make({3.0F, -4.0F}, {2});
  const Handle b = make({1.0F, -2.0F, 2.0F, 0.0F}, {2, 2});
  const std::array<const tr_tensor*, 2> self{a.t, b.t};
  std::array<tr_tensor*, 2> out{};
  ASSERT_EQ(tr_gen__foreach_norm_scalar(self.data(), 2, 2.0, -1, out.data(), 2),
            0)
      << tr_last_error();
  const Handle n0(out[0]);
  const Handle n1(out[1]);
  expect_floats(data_of(n0.t), {5.0F});
  expect_floats(data_of(n1.t), {3.0F});
  ASSERT_EQ(
      tr_gen__foreach_norm_scalar(self.data(), 2, kInfD, -1, out.data(), 2), 0)
      << tr_last_error();
  const Handle m0(out[0]);
  const Handle m1(out[1]);
  expect_floats(data_of(m0.t), {4.0F});
  expect_floats(data_of(m1.t), {2.0F});
}

TEST(GeneratedTranche11, ForeachNormRefusesAMisalignedOutArray) {
  const Handle a = make({3.0F, -4.0F}, {2});
  const Handle b = make({1.0F}, {1});
  const std::array<const tr_tensor*, 2> self{a.t, b.t};
  std::array<tr_tensor*, 2> out{a.t, b.t};
  expect_status_refused(
      tr_gen__foreach_norm_scalar(self.data(), 2, 2.0, -1, out.data(), 1),
      "tr_gen__foreach_norm_scalar");
  EXPECT_EQ(out[0], nullptr);
  out = {a.t, b.t};
  expect_status_refused(
      tr_gen__foreach_norm_scalar(nullptr, 2, 2.0, -1, out.data(), 2),
      "tr_gen__foreach_norm_scalar");
  EXPECT_EQ(out[0], nullptr);
  EXPECT_EQ(out[1], nullptr);
  expect_status_refused(
      tr_gen__foreach_norm_scalar(self.data(), 2, 2.0, -1, nullptr, 2),
      "tr_gen__foreach_norm_scalar");
  const std::array<const tr_tensor*, 2> holed{a.t, nullptr};
  out = {a.t, b.t};
  EXPECT_EQ(
      tr_gen__foreach_norm_scalar(holed.data(), 2, 2.0, -1, out.data(), 2), 1);
  EXPECT_EQ(out[0], nullptr);
  EXPECT_EQ(out[1], nullptr);
}

TEST(GeneratedTranche11, LinalgVectorNormReducesAllOrOneDim) {
  const Handle a = make({3.0F, 4.0F, 0.0F, -1.0F}, {2, 2});
  const Handle all(
      tr_gen_linalg_vector_norm(a.t, 1.0, nullptr, 0, false, false, -1));
  expect_floats(data_of(all.t), {8.0F});
  const std::array<int64_t, 1> dim{1};
  const Handle rows(
      tr_gen_linalg_vector_norm(a.t, 2.0, dim.data(), 1, true, false, -1));
  expect_floats(data_of(rows.t), {5.0F, 1.0F});
  EXPECT_EQ(
      tr_gen_linalg_vector_norm(nullptr, 2.0, nullptr, 0, false, false, -1),
      nullptr);
}

TEST(GeneratedTranche11, ForeachScaleAndClampUpdateEveryTensor) {
  const Handle a = make({1.0F, -3.0F}, {2});
  const Handle b = make({0.5F, 2.0F, -0.25F}, {3});
  const std::array<tr_tensor*, 2> self{a.t, b.t};
  const Handle half = scalar(0.5F);
  ASSERT_EQ(tr_gen__foreach_mul__tensor(self.data(), 2, half.t), 0)
      << tr_last_error();
  expect_floats(data_of(a.t), {0.5F, -1.5F});
  expect_floats(data_of(b.t), {0.25F, 1.0F, -0.125F});
  ASSERT_EQ(tr_gen__foreach_clamp_min__scalar(self.data(), 2, -0.5), 0);
  ASSERT_EQ(tr_gen__foreach_clamp_max__scalar(self.data(), 2, 0.5), 0);
  expect_floats(data_of(a.t), {0.5F, -0.5F});
  expect_floats(data_of(b.t), {0.25F, 0.5F, -0.125F});
  expect_status_refused(tr_gen__foreach_mul__tensor(self.data(), 2, nullptr),
                        "tr_gen__foreach_mul__tensor");
  expect_status_refused(tr_gen__foreach_clamp_min__scalar(nullptr, 1, 0.0),
                        "tr_gen__foreach_clamp_min__scalar");
}

TEST(GeneratedTranche11, AmpUnscaleFlagsANonFiniteGradient) {
  const Handle a = make({2.0F, -4.0F}, {2});
  const Handle b = make({kInf, 1.0F}, {2});
  const std::array<tr_tensor*, 2> self{a.t, b.t};
  const Handle found = scalar(0.0F);
  const Handle inv = scalar(0.5F);
  ASSERT_EQ(tr_gen__amp_foreach_non_finite_check_and_unscale_(self.data(), 1,
                                                              found.t, inv.t),
            0)
      << tr_last_error();
  expect_floats(data_of(a.t), {1.0F, -2.0F});
  expect_floats(data_of(found.t), {0.0F});
  ASSERT_EQ(tr_gen__amp_foreach_non_finite_check_and_unscale_(self.data(), 2,
                                                              found.t, inv.t),
            0)
      << tr_last_error();
  expect_floats(data_of(found.t), {1.0F});
  expect_status_refused(tr_gen__amp_foreach_non_finite_check_and_unscale_(
                            self.data(), 2, nullptr, inv.t),
                        "tr_gen__amp_foreach_non_finite_check_and_unscale_");
}

TEST(GeneratedTranche11, AmpUpdateScaleGrowsAndBacksOff) {
  const Handle scale = scalar(1024.0F);
  const Handle tracker = int32_scalar(1.0F);
  const Handle finite = scalar(0.0F);
  ASSERT_EQ(
      tr_gen__amp_update_scale_(scale.t, tracker.t, finite.t, 2.0, 0.5, 2), 0)
      << tr_last_error();
  expect_floats(data_of(scale.t), {2048.0F});
  expect_floats(data_of(tracker.t), {0.0F});
  const Handle inf = scalar(1.0F);
  ASSERT_EQ(tr_gen__amp_update_scale_(scale.t, tracker.t, inf.t, 2.0, 0.5, 2),
            0)
      << tr_last_error();
  expect_floats(data_of(scale.t), {1024.0F});
  expect_status_refused(
      tr_gen__amp_update_scale_(scale.t, nullptr, inf.t, 2.0, 0.5, 2),
      "tr_gen__amp_update_scale_");
}

}  // namespace
