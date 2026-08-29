#include <gtest/gtest.h>

#include <cmath>
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

std::vector<float> data_of(const tr_tensor* t) {
  int64_t numel = 0;
  EXPECT_EQ(tr_tensor_numel(t, &numel), 0) << tr_last_error();
  std::vector<float> out(static_cast<size_t>(numel));
  uint64_t got = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, out.size(), out.data(), &got), 0)
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

TEST(GeneratedAsr, Conv1dComputesMovingSums) {
  const Handle input = make({1.0F, 2.0F, 3.0F, 4.0F}, {1, 1, 4});
  const Handle weight = make({1.0F, 1.0F}, {1, 1, 2});
  const std::vector<int64_t> one{1};
  const std::vector<int64_t> zero{0};
  const Handle out(tr_gen_conv1d(input.t, weight.t, nullptr, one.data(), 1,
                                 zero.data(), 1, one.data(), 1, 1));
  EXPECT_EQ(shape_of(out.t), (std::vector<int64_t>{1, 1, 3}));
  EXPECT_EQ(data_of(out.t), (std::vector<float>{3.0F, 5.0F, 7.0F}));
  const Handle bias = make({10.0F}, {1});
  const Handle biased(tr_gen_conv1d(input.t, weight.t, bias.t, one.data(), 1,
                                    zero.data(), 1, one.data(), 1, 1));
  EXPECT_EQ(data_of(biased.t), (std::vector<float>{13.0F, 15.0F, 17.0F}));
  EXPECT_EQ(tr_gen_conv1d(nullptr, weight.t, nullptr, one.data(), 1,
                          zero.data(), 1, one.data(), 1, 1),
            nullptr);
  EXPECT_NE(tr_last_error(), nullptr);
}

TEST(GeneratedAsr, CtcLossMatchesClosedForm) {
  // Uniform log-probs over 2 classes for 2 frames; label "1" admits the
  // alignments [1 1], [blank 1], [1 blank] -> p = 3/4, loss = -ln(3/4).
  const float half = std::log(0.5F);
  const Handle log_probs = make({half, half, half, half}, {2, 1, 2});
  const Handle targets_f = make({1.0F}, {1, 1});
  const Handle targets(tr_tensor_to_dtype(targets_f.t, TR_DTYPE_INT64));
  const std::vector<int64_t> input_lengths{2};
  const std::vector<int64_t> target_lengths{1};
  const Handle loss(tr_gen_ctc_loss_intlist(
      log_probs.t, targets.t, input_lengths.data(), 1, target_lengths.data(), 1,
      /*blank=*/0, /*reduction=*/1, /*zero_infinity=*/false));
  EXPECT_EQ(shape_of(loss.t), std::vector<int64_t>{});
  EXPECT_NEAR(data_of(loss.t)[0], -std::log(0.75F), 1e-6);
  EXPECT_EQ(tr_gen_ctc_loss_intlist(nullptr, targets.t, input_lengths.data(), 1,
                                    target_lengths.data(), 1, 0, 1, false),
            nullptr);
  EXPECT_NE(tr_last_error(), nullptr);
}

}  // namespace
