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

void expect_error_from(const char* name) {
  EXPECT_NE(std::strstr(tr_last_error(), name), nullptr) << tr_last_error();
}

TEST(GeneratedTranche14, InplaceDivBroadcastsAPerChannelDivisor) {
  const Handle image = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 1, 3});
  const Handle std_dev = make({2.0F, 4.0F}, {2, 1, 1});
  EXPECT_EQ(tr_gen_div__tensor(image.t, std_dev.t), 0) << tr_last_error();
  EXPECT_EQ(data_of(image.t),
            (std::vector<float>{0.5F, 1.0F, 1.5F, 1.0F, 1.25F, 1.5F}));
}

TEST(GeneratedTranche14, InplaceDivRefusesAMissingOrUnbroadcastableDivisor) {
  const Handle image = make({1.0F, 2.0F, 3.0F}, {3});
  EXPECT_EQ(tr_gen_div__tensor(image.t, nullptr), 1);
  expect_error_from("tr_gen_div__tensor");
  EXPECT_EQ(tr_gen_div__tensor(nullptr, image.t), 1);
  expect_error_from("tr_gen_div__tensor");
  const Handle two = make({1.0F, 2.0F}, {2});
  EXPECT_NE(tr_gen_div__tensor(image.t, two.t), 0);
  EXPECT_EQ(data_of(image.t), (std::vector<float>{1.0F, 2.0F, 3.0F}));
}

}  // namespace
