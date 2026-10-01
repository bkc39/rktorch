#include <gtest/gtest.h>

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

void expect_near(const std::vector<float>& got, const std::vector<float>& want,
                 float tol) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
  }
}

struct Attention {
  Handle query = make({1.0F, 0.0F, 0.0F, 1.0F}, {2, 2});
  Handle key = make({1.0F, 0.0F, 0.0F, 1.0F}, {2, 2});
  Handle value = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});

  tr_tensor* operator()(const tr_tensor* mask, bool causal, double scale,
                        bool scale_has) const {
    return tr_gen_scaled_dot_product_attention(
        query.t, key.t, value.t, mask, 0.0, causal, scale, scale_has, false);
  }
};

// softmax([1/sqrt 2, 0]) = [0.669762, 0.330238] weighs the rows of value
const std::vector<float> kDefault{1.660477F, 2.660477F, 2.339523F, 3.339523F};
const std::vector<float> kCausal{1.0F, 2.0F, 2.339523F, 3.339523F};

TEST(GeneratedTranche8, AttentionScalesByOneOverRootEUnlessScaleIsPresent) {
  const Attention attend;
  const Handle absent(attend(nullptr, false, 0.0, false));
  expect_near(data_of(absent.t), kDefault, 1e-5F);
  // a present zero scale flattens the scores: every query averages the values
  const Handle flat(attend(nullptr, false, 0.0, true));
  expect_near(data_of(flat.t), {2.0F, 3.0F, 2.0F, 3.0F}, 1e-5F);
  const Handle explicit_default(attend(nullptr, false, 0.70710678, true));
  expect_near(data_of(explicit_default.t), kDefault, 1e-5F);
}

TEST(GeneratedTranche8, AttentionMasksCausallyByBoolOrByAddition) {
  const Attention attend;
  const Handle causal(attend(nullptr, true, 0.0, false));
  expect_near(data_of(causal.t), kCausal, 1e-5F);
  // a bool mask is true where a query may attend
  const Handle lower = make({1.0F, 0.0F, 1.0F, 1.0F}, {2, 2});
  const Handle allowed(tr_gen_ne_scalar(lower.t, 0.0));
  const Handle by_bool(attend(allowed.t, false, 0.0, false));
  expect_near(data_of(by_bool.t), kCausal, 1e-5F);
  const float hidden = -std::numeric_limits<float>::infinity();
  const Handle additive = make({0.0F, hidden, 0.0F, 0.0F}, {2, 2});
  const Handle by_float(attend(additive.t, false, 0.0, false));
  expect_near(data_of(by_float.t), kCausal, 1e-5F);
  EXPECT_EQ(attend(allowed.t, true, 0.0, false), nullptr);
  EXPECT_NE(std::strstr(tr_last_error(), "tr_gen_scaled_dot_product_attention"),
            nullptr);
}

TEST(GeneratedTranche8, AttentionRefusesAMissingTensor) {
  const Attention attend;
  EXPECT_EQ(tr_gen_scaled_dot_product_attention(nullptr, attend.key.t,
                                                attend.value.t, nullptr, 0.0,
                                                false, 0.0, false, false),
            nullptr);
  EXPECT_NE(std::strstr(tr_last_error(), "tr_gen_scaled_dot_product_attention"),
            nullptr);
  EXPECT_EQ(tr_gen_scaled_dot_product_attention(attend.query.t, attend.key.t,
                                                nullptr, nullptr, 0.0, false,
                                                0.0, false, false),
            nullptr);
}

}  // namespace
