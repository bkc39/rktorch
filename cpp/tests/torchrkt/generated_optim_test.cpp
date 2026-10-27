#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
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

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

TEST(TorchrktGeneratedOptim, ForeachLerpIsPerTensorLerpBitForBit) {
  Handle a = make({0.1F, -2.5F, 3.25F}, {3});
  Handle b = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  Handle a_end = make({7.0F, 0.3F, -1.0F}, {3});
  Handle b_end = make({-4.0F, 0.5F, 9.0F, 1e-3F}, {2, 2});
  Handle a_twin = make({0.1F, -2.5F, 3.25F}, {3});
  Handle b_twin = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const double weight = 0.1;

  std::vector<tr_tensor*> selves{a.t, b.t};
  std::vector<const tr_tensor*> ends{a_end.t, b_end.t};
  ASSERT_EQ(tr_gen__foreach_lerp__scalar(selves.data(), 2, ends.data(), 2,
                                         weight),
            0)
      << tr_last_error();
  ASSERT_EQ(tr_gen_lerp__scalar(a_twin.t, a_end.t, weight), 0)
      << tr_last_error();
  ASSERT_EQ(tr_gen_lerp__scalar(b_twin.t, b_end.t, weight), 0)
      << tr_last_error();
  EXPECT_TRUE(same_bits(data_of(a.t), data_of(a_twin.t)));
  EXPECT_TRUE(same_bits(data_of(b.t), data_of(b_twin.t)));
}

TEST(TorchrktGeneratedOptim, AnEmptyListReachesTorchAndANullListIsRefused) {
  Handle a = make({1.0F, 2.0F}, {2});
  EXPECT_EQ(tr_gen__foreach_add__scalar(nullptr, 0, 1.0), 1);
  EXPECT_NE(std::string(tr_last_error())
                .find("Tensor list must have at least one tensor"),
            std::string::npos)
      << tr_last_error();
  EXPECT_EQ(tr_gen__foreach_add__scalar(nullptr, 2, 1.0), 1);
  EXPECT_EQ(std::string(tr_last_error()),
            "tr_gen__foreach_add__scalar: null argument");
  std::vector<tr_tensor*> selves{a.t};
  EXPECT_EQ(tr_gen__foreach_div__scalarlist(selves.data(), 1, nullptr, 1), 1);
  EXPECT_EQ(data_of(a.t), (std::vector<float>{1.0F, 2.0F}));
  const std::vector<double> halves{0.5};
  ASSERT_EQ(tr_gen__foreach_div__scalarlist(selves.data(), 1, halves.data(), 1),
            0)
      << tr_last_error();
  EXPECT_EQ(data_of(a.t), (std::vector<float>{2.0F, 4.0F}));
}

}  // namespace
