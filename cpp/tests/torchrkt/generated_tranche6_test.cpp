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

TEST(GeneratedTranche6, BatchNormNormalisesPerChannelAndUpdatesRunningStats) {
  // two channels of two pixels each: {1, 3} and {5, 7}
  const Handle input = make({1.0F, 3.0F, 5.0F, 7.0F}, {1, 2, 1, 2});
  const Handle batch(tr_gen_batch_norm(input.t, nullptr, nullptr, nullptr,
                                       nullptr, true, 0.1, 1e-5, true));
  expect_near(data_of(batch.t), {-1.0F, 1.0F, -1.0F, 1.0F}, 1e-4F);
  const Handle mean = make({0.0F, 0.0F}, {2});
  const Handle var = make({1.0F, 1.0F}, {2});
  const Handle eval(tr_gen_batch_norm(input.t, nullptr, nullptr, mean.t, var.t,
                                      false, 0.1, 1e-5, true));
  expect_near(data_of(eval.t), {1.0F, 3.0F, 5.0F, 7.0F}, 1e-4F);
  // training with statistics updates them in place: momentum 0.1 of the
  // batch mean and of the unbiased batch variance (2 for both channels)
  const Handle tracked(tr_gen_batch_norm(input.t, nullptr, nullptr, mean.t,
                                         var.t, true, 0.1, 1e-5, true));
  expect_near(data_of(tracked.t), {-1.0F, 1.0F, -1.0F, 1.0F}, 1e-4F);
  expect_near(data_of(mean.t), {0.2F, 0.6F}, 1e-5F);
  expect_near(data_of(var.t), {1.1F, 1.1F}, 1e-5F);
  const Handle weight = make({2.0F, 2.0F}, {2});
  const Handle bias = make({1.0F, 1.0F}, {2});
  const Handle affine(tr_gen_batch_norm(input.t, weight.t, bias.t, nullptr,
                                        nullptr, true, 0.1, 1e-5, true));
  const std::vector<float> bare = data_of(batch.t);
  const std::vector<float> scaled = data_of(affine.t);
  for (size_t i = 0; i < bare.size(); ++i) {
    EXPECT_NEAR(scaled[i], 2.0F * bare[i] + 1.0F, 1e-4F);
  }
  // evaluation needs the statistics
  EXPECT_EQ(tr_gen_batch_norm(input.t, nullptr, nullptr, nullptr, nullptr,
                              false, 0.1, 1e-5, true),
            nullptr);
  expect_error_from("tr_gen_batch_norm");
  EXPECT_EQ(tr_gen_batch_norm(nullptr, nullptr, nullptr, nullptr, nullptr, true,
                              0.1, 1e-5, true),
            nullptr);
  expect_error_from("tr_gen_batch_norm");
}

TEST(GeneratedTranche6, LeakyReluScalesTheNegativeSide) {
  const Handle x = make({-2.0F, -0.5F, 0.0F, 3.0F}, {4});
  const Handle y(tr_gen_leaky_relu(x.t, 0.1));
  expect_near(data_of(y.t), {-0.2F, -0.05F, 0.0F, 3.0F}, 1e-6F);
  EXPECT_EQ(tr_gen_leaky_relu(nullptr, 0.1), nullptr);
  expect_error_from("tr_gen_leaky_relu");
}

TEST(GeneratedTranche6, BinaryCrossEntropyWithLogitsAtZeroIsLog2) {
  const Handle logits = make({0.0F, 0.0F}, {2});
  const Handle targets = make({1.0F, 0.0F}, {2});
  const float log2 = 0.69314718F;
  const Handle mean(tr_gen_binary_cross_entropy_with_logits(
      logits.t, targets.t, nullptr, nullptr, 1));
  expect_near(data_of(mean.t), {log2}, 1e-6F);
  const Handle sum(tr_gen_binary_cross_entropy_with_logits(
      logits.t, targets.t, nullptr, nullptr, 2));
  expect_near(data_of(sum.t), {2.0F * log2}, 1e-6F);
  const Handle weight = make({1.0F, 3.0F}, {2});
  const Handle weighted(tr_gen_binary_cross_entropy_with_logits(
      logits.t, targets.t, weight.t, nullptr, 1));
  expect_near(data_of(weighted.t), {2.0F * log2}, 1e-6F);
  const Handle none(tr_gen_binary_cross_entropy_with_logits(
      logits.t, targets.t, nullptr, nullptr, 0));
  EXPECT_EQ(shape_of(none.t), (std::vector<int64_t>{2}));
  EXPECT_EQ(tr_gen_binary_cross_entropy_with_logits(nullptr, targets.t, nullptr,
                                                    nullptr, 1),
            nullptr);
  expect_error_from("tr_gen_binary_cross_entropy_with_logits");
}

TEST(GeneratedTranche6, HuberLossIsQuadraticInsideDeltaLinearOutside) {
  const Handle x = make({0.0F, 0.0F, 0.0F}, {3});
  const Handle target = make({0.5F, 2.0F, -3.0F}, {3});
  // delta 1: 0.125, 1.5, 2.5
  const Handle unit(tr_gen_huber_loss(x.t, target.t, 1, 1.0));
  expect_near(data_of(unit.t), {1.375F}, 1e-6F);
  // delta 2: 0.125, 2, 4
  const Handle wide(tr_gen_huber_loss(x.t, target.t, 1, 2.0));
  expect_near(data_of(wide.t), {2.0416667F}, 1e-6F);
  const Handle none(tr_gen_huber_loss(x.t, target.t, 0, 1.0));
  expect_near(data_of(none.t), {0.125F, 1.5F, 2.5F}, 1e-6F);
  EXPECT_EQ(tr_gen_huber_loss(nullptr, target.t, 1, 1.0), nullptr);
  expect_error_from("tr_gen_huber_loss");
}

TEST(GeneratedTranche6, L1LossIsTheMeanAbsoluteError) {
  const Handle x = make({0.0F, 0.0F, 0.0F}, {3});
  const Handle target = make({0.5F, 2.0F, -3.0F}, {3});
  const Handle mean(tr_gen_l1_loss(x.t, target.t, 1));
  expect_near(data_of(mean.t), {1.8333333F}, 1e-6F);
  const Handle sum(tr_gen_l1_loss(x.t, target.t, 2));
  expect_near(data_of(sum.t), {5.5F}, 1e-6F);
  EXPECT_EQ(tr_gen_l1_loss(nullptr, target.t, 1), nullptr);
  expect_error_from("tr_gen_l1_loss");
}

TEST(GeneratedTranche6, FlipReversesTheNamedDims) {
  const Handle x = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const int64_t cols[] = {1};
  const Handle mirrored(tr_gen_flip(x.t, cols, 1));
  expect_near(data_of(mirrored.t), {2.0F, 1.0F, 4.0F, 3.0F}, 0.0F);
  const int64_t both[] = {0, 1};
  const Handle rotated(tr_gen_flip(x.t, both, 2));
  expect_near(data_of(rotated.t), {4.0F, 3.0F, 2.0F, 1.0F}, 0.0F);
  EXPECT_EQ(tr_gen_flip(nullptr, cols, 1), nullptr);
  expect_error_from("tr_gen_flip");
}

}  // namespace
