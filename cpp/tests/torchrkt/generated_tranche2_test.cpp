#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "torchrkt/c_api.h"

// C-boundary goldens for the tranche-2 generated families (#3): one
// correctness case + one error-path case per family. Value parity with
// PyTorch is the Racket python-cross-test's job; these pin the C contract
// the parity battery can't see — null/length guards, the int-status
// in-place shape, and the optional-int-array presence flag (and its guard).

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

// ---- compare: float32 mask + null guard --------------------------------

TEST(GeneratedTranche2, CompareTensorAndScalar) {
  const Handle a = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle b = make({1.0F, 5.0F, 3.0F}, {3});
  const Handle eq_t(tr_gen_eq_tensor(a.t, b.t));
  EXPECT_EQ(data_of(eq_t.t), (std::vector<float>{1.0F, 0.0F, 1.0F}));
  const Handle ge_s(tr_gen_ge_scalar(a.t, 2.0));
  EXPECT_EQ(data_of(ge_s.t), (std::vector<float>{0.0F, 1.0F, 1.0F}));
  EXPECT_EQ(tr_gen_eq_tensor(nullptr, b.t), nullptr);
  expect_error_from("tr_gen_eq_tensor");
}

// ---- conv: shape + optional-tensor bias (NULL) + null guard ------------

TEST(GeneratedTranche2, Conv2dShapeWithAndWithoutBias) {
  // 1x1x3x3 input, one 1x1x2x2 filter, stride/dilation 1, no padding -> 2x2.
  const Handle input = make({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
  const Handle weight = make({1, 0, 0, 1}, {1, 1, 2, 2});
  const std::vector<int64_t> one = {1, 1};
  const std::vector<int64_t> zero = {0, 0};
  const Handle no_bias(tr_gen_conv2d(input.t, weight.t, nullptr, one.data(), 2,
                                     zero.data(), 2, one.data(), 2, 1));
  EXPECT_EQ(shape_of(no_bias.t), (std::vector<int64_t>{1, 1, 2, 2}));
  const Handle bias = make({10.0F}, {1});
  const Handle with_bias(tr_gen_conv2d(input.t, weight.t, bias.t, one.data(), 2,
                                       zero.data(), 2, one.data(), 2, 1));
  // bias adds 10 to every output cell vs the no-bias result.
  const std::vector<float> base = data_of(no_bias.t);
  std::vector<float> biased = data_of(with_bias.t);
  for (size_t i = 0; i < biased.size(); ++i) {
    EXPECT_FLOAT_EQ(biased[i], base[i] + 10.0F);
  }
  EXPECT_EQ(tr_gen_conv2d(nullptr, weight.t, nullptr, one.data(), 2,
                          zero.data(), 2, one.data(), 2, 1),
            nullptr);
  expect_error_from("tr_gen_conv2d");
}

// ---- pooling family: shape/value + null guards -------------------------

TEST(GeneratedTranche2, PoolingFamilyGoldens) {
  // 1x1x2x2 input {1,2,3,4}; a single 2x2 window.
  const Handle in = make({1.0F, 2.0F, 3.0F, 4.0F}, {1, 1, 2, 2});
  const std::vector<int64_t> k = {2, 2};
  const std::vector<int64_t> z = {0, 0};
  const std::vector<int64_t> one = {1, 1};

  const Handle mp(tr_gen_max_pool2d(in.t, k.data(), 2, k.data(), 2, z.data(), 2,
                                    one.data(), 2, /*ceil=*/false));
  EXPECT_EQ(shape_of(mp.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(data_of(mp.t), (std::vector<float>{4.0F}));  // max

  const Handle ap(tr_gen_avg_pool2d(in.t, k.data(), 2, k.data(), 2, z.data(), 2,
                                    /*ceil=*/false, /*count_include_pad=*/true,
                                    /*divisor_override=*/0, /*has=*/false));
  EXPECT_EQ(shape_of(ap.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_NEAR(data_of(ap.t).at(0), 2.5F, 1e-5F);  // mean of 1,2,3,4

  const Handle aap(tr_gen_adaptive_avg_pool2d(in.t, one.data(), 2));
  EXPECT_EQ(shape_of(aap.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_NEAR(data_of(aap.t).at(0), 2.5F, 1e-5F);  // global avg

  // null self and null int-array both surface as NULL + error.
  EXPECT_EQ(tr_gen_max_pool2d(nullptr, k.data(), 2, k.data(), 2, z.data(), 2,
                              one.data(), 2, false),
            nullptr);
  expect_error_from("tr_gen_max_pool2d");
  EXPECT_EQ(tr_gen_avg_pool2d(in.t, nullptr, 2, k.data(), 2, z.data(), 2, false,
                              true, 0, false),
            nullptr);
  expect_error_from("tr_gen_avg_pool2d");
  EXPECT_EQ(tr_gen_adaptive_avg_pool2d(nullptr, one.data(), 2), nullptr);
  expect_error_from("tr_gen_adaptive_avg_pool2d");
}

// ---- in-place (int-status shape): mutate self, status, null guard ------

TEST(GeneratedTranche2, InplaceMulMutatesAndStatus) {
  const Handle a = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle b = make({2.0F, 3.0F, 4.0F}, {3});
  EXPECT_EQ(tr_gen_mul__tensor(a.t, b.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{2.0F, 6.0F, 12.0F}));
  // add_ alpha form: a += 10 * b  (a is now {2,6,12}).
  EXPECT_EQ(tr_gen_add__tensor(a.t, b.t, 10.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{22.0F, 36.0F, 52.0F}));
  // The in-place guard returns the int-status failure, not a crash.
  EXPECT_EQ(tr_gen_mul__tensor(a.t, nullptr), 1);
  expect_error_from("tr_gen_mul__tensor");
}

// ---- loss: int64 targets, scalar output, null guard --------------------

TEST(GeneratedTranche2, NllLossScalarOutput) {
  // 2 samples, 3 classes; log-probs already (nll_loss expects log-probs).
  const Handle logp = make({-0.5F, -1.0F, -2.0F, -2.0F, -0.2F, -1.5F}, {2, 3});
  const Handle target_f = make({0.0F, 1.0F}, {2});
  const Handle target(tr_tensor_to_dtype(target_f.t, TR_DTYPE_INT64));
  const Handle loss(tr_gen_nll_loss(logp.t, target.t, nullptr, /*mean=*/1,
                                    /*ignore_index=*/-100));
  EXPECT_EQ(shape_of(loss.t), (std::vector<int64_t>{}));  // scalar
  // mean(-(-0.5), -(-0.2)) = mean(0.5, 0.2) = 0.35
  EXPECT_NEAR(data_of(loss.t).at(0), 0.35F, 1e-5F);
  // optional-tensor weight present: weighted mean divides by the summed
  // weights of the targets. w=(2,3,4), targets (0,1):
  // (2*0.5 + 3*0.2) / (2+3) = 1.6/5 = 0.32.
  const Handle weight = make({2.0F, 3.0F, 4.0F}, {3});
  const Handle wloss(tr_gen_nll_loss(logp.t, target.t, weight.t, 1, -100));
  EXPECT_NEAR(data_of(wloss.t).at(0), 0.32F, 1e-5F);
  EXPECT_EQ(tr_gen_nll_loss(nullptr, target.t, nullptr, 1, -100), nullptr);
  expect_error_from("tr_gen_nll_loss");
}

// ---- shape: narrow (tensor + three int64 scalars) ----------------------

TEST(GeneratedTranche2, NarrowSlicesAndGuards) {
  const Handle a = make({10.0F, 20.0F, 30.0F, 40.0F}, {4});
  const Handle s(tr_gen_narrow(a.t, /*dim=*/0, /*start=*/1, /*length=*/2));
  EXPECT_EQ(shape_of(s.t), (std::vector<int64_t>{2}));
  EXPECT_EQ(data_of(s.t), (std::vector<float>{20.0F, 30.0F}));
  EXPECT_EQ(tr_gen_narrow(nullptr, 0, 0, 1), nullptr);
  expect_error_from("tr_gen_narrow");
}

// ---- reduce (optional-int-array presence flag + its guard) -------------

TEST(GeneratedTranche2, SumDimPresenceFlagAndGuard) {
  const Handle a = make({1, 2, 3, 4, 5, 6}, {2, 3});
  const std::vector<int64_t> dim = {1};
  // dim present: sum along dim 1 -> {6, 15}.
  const Handle along(tr_gen_sum_dim_intlist(a.t, dim.data(), 1,
                                            /*dim_has=*/true,
                                            /*keepdim=*/false, /*dtype=*/-1));
  EXPECT_EQ(shape_of(along.t), (std::vector<int64_t>{2}));
  EXPECT_EQ(data_of(along.t), (std::vector<float>{6.0F, 15.0F}));
  // dim absent: full sum -> scalar 21.
  const Handle full(
      tr_gen_sum_dim_intlist(a.t, nullptr, 0, /*dim_has=*/false, false, -1));
  EXPECT_NEAR(data_of(full.t).at(0), 21.0F, 1e-5F);
  // present-but-null is rejected by the guard (was UB before the fix).
  EXPECT_EQ(
      tr_gen_sum_dim_intlist(a.t, nullptr, 1, /*dim_has=*/true, false, -1),
      nullptr);
  expect_error_from("tr_gen_sum_dim_intlist");
}

}  // namespace
