#include <gtest/gtest.h>

#include <cmath>
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

TEST(GeneratedTranche2, Conv2dShapeWithAndWithoutBias) {
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

TEST(GeneratedTranche2, PoolingFamilyGoldens) {
  const Handle in = make({1.0F, 2.0F, 3.0F, 4.0F}, {1, 1, 2, 2});
  const std::vector<int64_t> k = {2, 2};
  const std::vector<int64_t> z = {0, 0};
  const std::vector<int64_t> one = {1, 1};

  const Handle mp(tr_gen_max_pool2d(in.t, k.data(), 2, k.data(), 2, z.data(), 2,
                                    one.data(), 2, /*ceil=*/false));
  EXPECT_EQ(shape_of(mp.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(data_of(mp.t), (std::vector<float>{4.0F}));

  const Handle mp_empty(tr_gen_max_pool2d(in.t, k.data(), 2, k.data(), 0,
                                          z.data(), 2, one.data(), 2, false));
  EXPECT_EQ(shape_of(mp_empty.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(data_of(mp_empty.t), (std::vector<float>{4.0F}));

  const Handle ap(tr_gen_avg_pool2d(in.t, k.data(), 2, k.data(), 2, z.data(), 2,
                                    /*ceil=*/false, /*count_include_pad=*/true,
                                    /*divisor_override=*/0, /*has=*/false));
  EXPECT_EQ(shape_of(ap.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_NEAR(data_of(ap.t).at(0), 2.5F, 1e-5F);

  const Handle aap(tr_gen_adaptive_avg_pool2d(in.t, one.data(), 2));
  EXPECT_EQ(shape_of(aap.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_NEAR(data_of(aap.t).at(0), 2.5F, 1e-5F);

  const Handle ap_div(tr_gen_avg_pool2d(in.t, k.data(), 2, k.data(), 2,
                                        z.data(), 2, /*ceil=*/false,
                                        /*count_include_pad=*/true,
                                        /*divisor_override=*/2, /*has=*/true));
  EXPECT_NEAR(data_of(ap_div.t).at(0), 5.0F, 1e-5F);

  // empty stride (len 0) is ATen's default-to-kernel_size passthrough
  const Handle ap_empty(tr_gen_avg_pool2d(in.t, k.data(), 2, k.data(), 0,
                                          z.data(), 2, false, true, 0, false));
  EXPECT_EQ(shape_of(ap_empty.t), (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_NEAR(data_of(ap_empty.t).at(0), 2.5F, 1e-5F);

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

TEST(GeneratedTranche2, DropoutEvalIdentityTrainScales) {
  const Handle x = make({1.0F, 2.0F, 3.0F, 4.0F}, {4});
  const Handle ev(tr_gen_dropout(x.t, 0.5, /*train=*/false));
  EXPECT_EQ(data_of(ev.t), (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}));
  const Handle tr0(tr_gen_dropout(x.t, 0.0, /*train=*/true));
  EXPECT_EQ(data_of(tr0.t), (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}));
  // inverted dropout: kept entries scale by 1/(1-p) = 2
  tr_manual_seed(0);
  const Handle tr(tr_gen_dropout(x.t, 0.5, /*train=*/true));
  const std::vector<float> in = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> out = data_of(tr.t);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_TRUE(out[i] == 0.0F || out[i] == 2.0F * in[i]) << out[i];
  }
  EXPECT_EQ(tr_gen_dropout(nullptr, 0.5, false), nullptr);
  expect_error_from("tr_gen_dropout");
}

TEST(GeneratedTranche2, IndexedWriteFamilyGoldens) {
  const Handle m = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3});
  const std::vector<int64_t> iv = {1};
  const std::vector<int64_t> id1 = {1};
  const Handle idx(tr_from_data_i64(iv.data(), iv.size(), id1.data(), 1));
  const Handle src = make({9.0F, 9.0F, 9.0F}, {1, 3});
  EXPECT_EQ(tr_gen_index_copy_(m.t, 0, idx.t, src.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(m.t),
            (std::vector<float>{1.0F, 2.0F, 3.0F, 9.0F, 9.0F, 9.0F}));
  EXPECT_EQ(tr_gen_index_add_(m.t, 0, idx.t, src.t, 2.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(m.t),
            (std::vector<float>{1.0F, 2.0F, 3.0F, 27.0F, 27.0F, 27.0F}));
  EXPECT_EQ(tr_gen_index_fill__int_scalar(m.t, 0, idx.t, 0.0), 0)
      << tr_last_error();
  const Handle mask(tr_gen_gt_scalar(m.t, 2.0));
  EXPECT_EQ(tr_gen_masked_fill__scalar(m.t, mask.t, -1.0), 0)
      << tr_last_error();
  EXPECT_EQ(data_of(m.t),
            (std::vector<float>{1.0F, 2.0F, -1.0F, 0.0F, 0.0F, 0.0F}));
  const std::vector<int64_t> sv = {0, 2, 1, 0};
  const std::vector<int64_t> sd = {2, 2};
  const Handle sidx(tr_from_data_i64(sv.data(), sv.size(), sd.data(), 2));
  EXPECT_EQ(tr_gen_scatter__value(m.t, 1, sidx.t, 7.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(m.t),
            (std::vector<float>{7.0F, 2.0F, 7.0F, 7.0F, 7.0F, 0.0F}));
  EXPECT_EQ(tr_gen_index_copy_(nullptr, 0, idx.t, src.t), 1);
  expect_error_from("tr_gen_index_copy_");
  EXPECT_EQ(tr_gen_masked_scatter_(m.t, nullptr, src.t), 1);
  expect_error_from("tr_gen_masked_scatter_");
  EXPECT_EQ(tr_gen_index_add_(nullptr, 0, idx.t, src.t, 1.0), 1);
  expect_error_from("tr_gen_index_add_");
  EXPECT_EQ(tr_gen_index_fill__int_scalar(nullptr, 0, idx.t, 0.0), 1);
  expect_error_from("tr_gen_index_fill__int_scalar");
  EXPECT_EQ(tr_gen_masked_fill__scalar(nullptr, mask.t, 0.0), 1);
  expect_error_from("tr_gen_masked_fill__scalar");
  EXPECT_EQ(tr_gen_scatter__value(nullptr, 1, sidx.t, 0.0), 1);
  expect_error_from("tr_gen_scatter__value");
}

TEST(GeneratedTranche2, IndexedWriteFamilyGoldensTwo) {
  const Handle m = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  EXPECT_EQ(tr_gen_fill__scalar(m.t, 9.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(m.t), (std::vector<float>{9.0F, 9.0F, 9.0F, 9.0F}));
  EXPECT_EQ(tr_gen_fill__scalar(nullptr, 9.0), 1);
  expect_error_from("tr_gen_fill__scalar");

  const std::vector<int64_t> iv = {1};
  const std::vector<int64_t> id1 = {1};
  const Handle idx(tr_from_data_i64(iv.data(), iv.size(), id1.data(), 1));
  const std::vector<float> vv = {5.0F};
  const std::vector<int64_t> vd = {};
  const Handle val(tr_from_data(vv.data(), 1, vd.data(), 0));
  EXPECT_EQ(tr_gen_index_fill__int_tensor(m.t, 0, idx.t, val.t), 0)
      << tr_last_error();
  EXPECT_EQ(data_of(m.t), (std::vector<float>{9.0F, 9.0F, 5.0F, 5.0F}));
  EXPECT_EQ(tr_gen_index_fill__int_tensor(nullptr, 0, idx.t, val.t), 1);
  expect_error_from("tr_gen_index_fill__int_tensor");

  const Handle mask(tr_gen_gt_scalar(m.t, 8.0));
  const std::vector<float> zv = {0.0F};
  const Handle zero(tr_from_data(zv.data(), 1, vd.data(), 0));
  EXPECT_EQ(tr_gen_masked_fill__tensor(m.t, mask.t, zero.t), 0)
      << tr_last_error();
  EXPECT_EQ(data_of(m.t), (std::vector<float>{0.0F, 0.0F, 5.0F, 5.0F}));
  EXPECT_EQ(tr_gen_masked_fill__tensor(m.t, nullptr, zero.t), 1);
  expect_error_from("tr_gen_masked_fill__tensor");

  const std::vector<int64_t> sv = {0, 1};
  const std::vector<int64_t> sd = {2, 1};
  const Handle sidx(tr_from_data_i64(sv.data(), sv.size(), sd.data(), 2));
  const Handle ssrc = make({70.0F, 80.0F}, {2, 1});
  EXPECT_EQ(tr_gen_scatter__src(m.t, 1, sidx.t, ssrc.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(m.t), (std::vector<float>{70.0F, 0.0F, 5.0F, 80.0F}));
  EXPECT_EQ(tr_gen_scatter__src(nullptr, 1, sidx.t, ssrc.t), 1);
  expect_error_from("tr_gen_scatter__src");

  EXPECT_EQ(tr_gen_scatter_add_(m.t, 1, sidx.t, ssrc.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(m.t), (std::vector<float>{140.0F, 0.0F, 5.0F, 160.0F}));
  EXPECT_EQ(tr_gen_scatter_add_(m.t, 1, nullptr, ssrc.t), 1);
  expect_error_from("tr_gen_scatter_add_");
}

TEST(GeneratedTranche2, InplaceCopyOverwritesSelf) {
  const Handle a = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle src = make({4.0F, 5.0F, 6.0F}, {3});
  EXPECT_EQ(tr_gen_copy_(a.t, src.t, /*non_blocking=*/false), 0)
      << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{4.0F, 5.0F, 6.0F}));
  EXPECT_EQ(tr_gen_copy_(a.t, nullptr, false), 1);
  expect_error_from("tr_gen_copy_");
}

TEST(GeneratedTranche2, InplaceMulMutatesAndStatus) {
  const Handle a = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle b = make({2.0F, 3.0F, 4.0F}, {3});
  EXPECT_EQ(tr_gen_mul__tensor(a.t, b.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{2.0F, 6.0F, 12.0F}));
  EXPECT_EQ(tr_gen_add__tensor(a.t, b.t, 10.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{22.0F, 36.0F, 52.0F}));
  EXPECT_EQ(tr_gen_mul__tensor(a.t, nullptr), 1);
  expect_error_from("tr_gen_mul__tensor");
}

TEST(GeneratedTranche2, InplaceThreeTensorAndAddcShapes) {
  const Handle l = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle end = make({5.0F, 6.0F, 7.0F}, {3});
  const Handle w = make({0.5F, 0.5F, 0.5F}, {3});
  EXPECT_EQ(tr_gen_lerp__tensor(l.t, end.t, w.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(l.t), (std::vector<float>{3.0F, 4.0F, 5.0F}));
  EXPECT_EQ(tr_gen_lerp__tensor(l.t, nullptr, w.t), 1);
  expect_error_from("tr_gen_lerp__tensor");

  const Handle cm = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle t1 = make({2.0F, 2.0F, 2.0F}, {3});
  const Handle t2 = make({3.0F, 3.0F, 3.0F}, {3});
  EXPECT_EQ(tr_gen_addcmul_(cm.t, t1.t, t2.t, 2.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(cm.t), (std::vector<float>{13.0F, 14.0F, 15.0F}));
  EXPECT_EQ(tr_gen_addcmul_(cm.t, nullptr, t2.t, 2.0), 1);
  expect_error_from("tr_gen_addcmul_");

  const Handle cd = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle n1 = make({4.0F, 4.0F, 4.0F}, {3});
  const Handle n2 = make({2.0F, 2.0F, 2.0F}, {3});
  EXPECT_EQ(tr_gen_addcdiv_(cd.t, n1.t, n2.t, 2.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(cd.t), (std::vector<float>{5.0F, 6.0F, 7.0F}));
  EXPECT_EQ(tr_gen_addcdiv_(cd.t, nullptr, n2.t, 2.0), 1);
  expect_error_from("tr_gen_addcdiv_");
}

TEST(GeneratedTranche2, NllLossScalarOutput) {
  const Handle logp = make({-0.5F, -1.0F, -2.0F, -2.0F, -0.2F, -1.5F}, {2, 3});
  const Handle target_f = make({0.0F, 1.0F}, {2});
  const Handle target(tr_tensor_to_dtype(target_f.t, TR_DTYPE_INT64));
  const Handle loss(tr_gen_nll_loss(logp.t, target.t, nullptr, /*mean=*/1,
                                    /*ignore_index=*/-100));
  EXPECT_EQ(shape_of(loss.t), (std::vector<int64_t>{}));
  EXPECT_NEAR(data_of(loss.t).at(0), 0.35F, 1e-5F);
  // weighted nll divides by the summed target weights:
  // (2*0.5 + 3*0.2) / (2+3) = 0.32
  const Handle weight = make({2.0F, 3.0F, 4.0F}, {3});
  const Handle wloss(tr_gen_nll_loss(logp.t, target.t, weight.t, 1, -100));
  EXPECT_NEAR(data_of(wloss.t).at(0), 0.32F, 1e-5F);
  EXPECT_EQ(tr_gen_nll_loss(nullptr, target.t, nullptr, 1, -100), nullptr);
  expect_error_from("tr_gen_nll_loss");
}

TEST(GeneratedTranche2, CrossEntropyLossGoldenAndGuard) {
  const Handle logits =
      make({-0.5F, -1.0F, -2.0F, -2.0F, -0.2F, -1.5F}, {2, 3});
  const Handle target_f = make({0.0F, 1.0F}, {2});
  const Handle target(tr_tensor_to_dtype(target_f.t, TR_DTYPE_INT64));
  const Handle loss(tr_gen_cross_entropy_loss(logits.t, target.t, nullptr,
                                              /*mean=*/1, /*ignore_index=*/-100,
                                              /*label_smoothing=*/0.0));
  EXPECT_EQ(shape_of(loss.t), (std::vector<int64_t>{}));
  // row0 -log_softmax[0]=0.60413, row1 -log_softmax[1]=0.36311, mean 0.48362
  EXPECT_NEAR(data_of(loss.t).at(0), 0.48362F, 1e-4F);
  const Handle weight = make({2.0F, 3.0F, 4.0F}, {3});
  const Handle wloss(
      tr_gen_cross_entropy_loss(logits.t, target.t, weight.t, 1, -100, 0.0));
  EXPECT_TRUE(std::isfinite(data_of(wloss.t).at(0)));
  EXPECT_NE(data_of(wloss.t).at(0), data_of(loss.t).at(0));
  const Handle smoothed(tr_gen_cross_entropy_loss(logits.t, target.t, nullptr,
                                                  1, -100, /*smoothing=*/0.5));
  EXPECT_GT(data_of(smoothed.t).at(0), data_of(loss.t).at(0));
  EXPECT_EQ(tr_gen_cross_entropy_loss(nullptr, target.t, nullptr, 1, -100, 0.0),
            nullptr);
  expect_error_from("tr_gen_cross_entropy_loss");
}

TEST(GeneratedTranche2, NarrowSlicesAndGuards) {
  const Handle a = make({10.0F, 20.0F, 30.0F, 40.0F}, {4});
  const Handle s(tr_gen_narrow(a.t, /*dim=*/0, /*start=*/1, /*length=*/2));
  EXPECT_EQ(shape_of(s.t), (std::vector<int64_t>{2}));
  EXPECT_EQ(data_of(s.t), (std::vector<float>{20.0F, 30.0F}));
  EXPECT_EQ(tr_tensor_sub_(s.t, s.t, 1.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{10.0F, 0.0F, 0.0F, 40.0F}));
  EXPECT_EQ(tr_gen_narrow(nullptr, 0, 0, 1), nullptr);
  expect_error_from("tr_gen_narrow");
}

TEST(GeneratedTranche2, SelectViewsAndGuards) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3});
  const Handle row(tr_gen_select_int(a.t, /*dim=*/0, /*index=*/1));
  EXPECT_EQ(shape_of(row.t), (std::vector<int64_t>{3}));
  EXPECT_EQ(data_of(row.t), (std::vector<float>{4.0F, 5.0F, 6.0F}));
  const Handle last(tr_gen_select_int(a.t, /*dim=*/1, /*index=*/-1));
  EXPECT_EQ(data_of(last.t), (std::vector<float>{3.0F, 6.0F}));
  EXPECT_EQ(tr_tensor_sub_(row.t, row.t, 1.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t),
            (std::vector<float>{1.0F, 2.0F, 3.0F, 0.0F, 0.0F, 0.0F}));
  EXPECT_EQ(tr_gen_select_int(nullptr, 0, 0), nullptr);
  expect_error_from("tr_gen_select_int");
  EXPECT_EQ(tr_gen_select_int(a.t, 0, 5), nullptr);
  expect_error_from("tr_gen_select_int");
}

TEST(GeneratedTranche2, SliceBoundsViewsAndGuards) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {6});
  const Handle s(tr_gen_slice_tensor(a.t, 0, 1, true, 5, true, 2));
  EXPECT_EQ(data_of(s.t), (std::vector<float>{2.0F, 4.0F}));
  const Handle open(tr_gen_slice_tensor(a.t, 0, 0, false, 0, false, 1));
  EXPECT_EQ(shape_of(open.t), (std::vector<int64_t>{6}));
  EXPECT_EQ(tr_tensor_sub_(s.t, s.t, 1.0), 0) << tr_last_error();
  EXPECT_EQ(data_of(a.t),
            (std::vector<float>{1.0F, 0.0F, 3.0F, 0.0F, 5.0F, 6.0F}));
  EXPECT_EQ(tr_gen_slice_tensor(nullptr, 0, 0, false, 0, false, 1), nullptr);
  expect_error_from("tr_gen_slice_tensor");
  EXPECT_EQ(tr_gen_slice_tensor(a.t, 0, 0, false, 0, false, -1), nullptr);
  expect_error_from("tr_gen_slice_tensor");
}

TEST(GeneratedTranche2, IndexSelectMaskedSelectAndGuards) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F}, {4});
  const std::vector<int64_t> idx_vals = {3, 0};
  const std::vector<int64_t> idx_dims = {2};
  const Handle idx(
      tr_from_data_i64(idx_vals.data(), idx_vals.size(), idx_dims.data(), 1));
  const Handle picked(tr_gen_index_select(a.t, 0, idx.t));
  EXPECT_EQ(data_of(picked.t), (std::vector<float>{4.0F, 1.0F}));
  EXPECT_EQ(tr_gen_index_select(nullptr, 0, idx.t), nullptr);
  expect_error_from("tr_gen_index_select");

  const Handle nz(tr_gen_nonzero(a.t));
  EXPECT_EQ(shape_of(nz.t), (std::vector<int64_t>{4, 1}));
  EXPECT_EQ(tr_gen_nonzero(nullptr), nullptr);
  expect_error_from("tr_gen_nonzero");

  const Handle taken(tr_gen_take(a.t, idx.t));
  EXPECT_EQ(data_of(taken.t), (std::vector<float>{4.0F, 1.0F}));
  const std::vector<int64_t> neg_vals = {-1, 0};
  const Handle neg_idx(
      tr_from_data_i64(neg_vals.data(), neg_vals.size(), idx_dims.data(), 1));
  const Handle neg_taken(tr_gen_take(a.t, neg_idx.t));
  EXPECT_EQ(data_of(neg_taken.t), (std::vector<float>{4.0F, 1.0F}));
  EXPECT_EQ(tr_gen_take(nullptr, idx.t), nullptr);
  expect_error_from("tr_gen_take");

  const Handle mask(tr_gen_gt_scalar(a.t, 2.0));
  const Handle kept(tr_gen_masked_select(a.t, mask.t));
  EXPECT_EQ(data_of(kept.t), (std::vector<float>{3.0F, 4.0F}));
  EXPECT_EQ(tr_gen_masked_select(a.t, nullptr), nullptr);
  expect_error_from("tr_gen_masked_select");
}

TEST(GeneratedTranche2, GatherTakeAlongWhereAndGuards) {
  const Handle m = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const std::vector<int64_t> iv = {1, 0, 0, 0};
  const std::vector<int64_t> id2 = {2, 2};
  const Handle idx(tr_from_data_i64(iv.data(), iv.size(), id2.data(), 2));
  const Handle g(tr_gen_gather(m.t, 1, idx.t, false));
  EXPECT_EQ(data_of(g.t), (std::vector<float>{2.0F, 1.0F, 3.0F, 3.0F}));
  EXPECT_EQ(tr_gen_gather(nullptr, 1, idx.t, false), nullptr);
  expect_error_from("tr_gen_gather");

  const Handle tad(tr_gen_take_along_dim(m.t, idx.t, 1, true));
  EXPECT_EQ(data_of(tad.t), (std::vector<float>{2.0F, 1.0F, 3.0F, 3.0F}));
  const Handle tad_flat(tr_gen_take_along_dim(m.t, idx.t, 0, false));
  EXPECT_EQ(shape_of(tad_flat.t), (std::vector<int64_t>{4}));
  EXPECT_EQ(data_of(tad_flat.t), (std::vector<float>{2.0F, 1.0F, 1.0F, 1.0F}));
  EXPECT_EQ(tr_gen_take_along_dim(nullptr, idx.t, 0, false), nullptr);
  expect_error_from("tr_gen_take_along_dim");

  const Handle cond(tr_gen_gt_scalar(m.t, 2.0));
  const Handle other = make({-1.0F, -1.0F, -1.0F, -1.0F}, {2, 2});
  const Handle ws(tr_gen_where_self(cond.t, m.t, other.t));
  EXPECT_EQ(data_of(ws.t), (std::vector<float>{-1.0F, -1.0F, 3.0F, 4.0F}));
  const Handle wso(tr_gen_where_scalarother(cond.t, m.t, 0.0));
  EXPECT_EQ(data_of(wso.t), (std::vector<float>{0.0F, 0.0F, 3.0F, 4.0F}));
  const Handle wss(tr_gen_where_scalarself(cond.t, 9.0, m.t));
  EXPECT_EQ(data_of(wss.t), (std::vector<float>{1.0F, 2.0F, 9.0F, 9.0F}));
  const Handle wsc(tr_gen_where_scalar(cond.t, 1.0, 0.0));
  EXPECT_EQ(data_of(wsc.t), (std::vector<float>{0.0F, 0.0F, 1.0F, 1.0F}));
  EXPECT_EQ(tr_gen_where_scalar(nullptr, 1.0, 0.0), nullptr);
  expect_error_from("tr_gen_where_scalar");
  EXPECT_EQ(tr_gen_where_scalarself(nullptr, 9.0, m.t), nullptr);
  expect_error_from("tr_gen_where_scalarself");
  EXPECT_EQ(tr_gen_where_self(nullptr, m.t, other.t), nullptr);
  expect_error_from("tr_gen_where_self");
  EXPECT_EQ(tr_gen_where_scalarother(cond.t, nullptr, 0.0), nullptr);
  expect_error_from("tr_gen_where_scalarother");
}

TEST(GeneratedTranche2, SumDimPresenceFlagAndGuard) {
  const Handle a = make({1, 2, 3, 4, 5, 6}, {2, 3});
  const std::vector<int64_t> dim = {1};
  const Handle along(tr_gen_sum_dim_intlist(a.t, dim.data(), 1,
                                            /*dim_has=*/true,
                                            /*keepdim=*/false, /*dtype=*/-1));
  EXPECT_EQ(shape_of(along.t), (std::vector<int64_t>{2}));
  EXPECT_EQ(data_of(along.t), (std::vector<float>{6.0F, 15.0F}));
  const Handle full(
      tr_gen_sum_dim_intlist(a.t, nullptr, 0, /*dim_has=*/false, false, -1));
  EXPECT_NEAR(data_of(full.t).at(0), 21.0F, 1e-5F);
  EXPECT_EQ(
      tr_gen_sum_dim_intlist(a.t, nullptr, 1, /*dim_has=*/true, false, -1),
      nullptr);
  expect_error_from("tr_gen_sum_dim_intlist");
  EXPECT_EQ(
      tr_gen_sum_dim_intlist(a.t, dim.data(), 0, /*dim_has=*/true, false, -1),
      nullptr);
  expect_error_from("tr_gen_sum_dim_intlist");
}

TEST(GeneratedTranche2, MeanDimPresenceFlagAndGuard) {
  const Handle a = make({1, 2, 3, 4, 5, 6}, {2, 3});
  const std::vector<int64_t> dim = {1};
  const Handle along(tr_gen_mean_dim(a.t, dim.data(), 1, /*dim_has=*/true,
                                     /*keepdim=*/false, /*dtype=*/-1));
  EXPECT_EQ(shape_of(along.t), (std::vector<int64_t>{2}));
  EXPECT_EQ(data_of(along.t), (std::vector<float>{2.0F, 5.0F}));
  const Handle full(
      tr_gen_mean_dim(a.t, nullptr, 0, /*dim_has=*/false, false, -1));
  EXPECT_NEAR(data_of(full.t).at(0), 3.5F, 1e-5F);
  EXPECT_EQ(tr_gen_mean_dim(a.t, nullptr, 1, /*dim_has=*/true, false, -1),
            nullptr);
  expect_error_from("tr_gen_mean_dim");
}

}  // namespace
