#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "torchrkt/c_api.h"

namespace {

tr_tensor* make_randn(const std::vector<int64_t>& shape) {
  tr_tensor* t = tr_randn(shape.data(), static_cast<int64_t>(shape.size()));
  EXPECT_NE(t, nullptr) << tr_last_error();
  return t;
}

std::vector<float> data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, 0, nullptr, &numel), 2) << tr_last_error();
  std::vector<float> out(numel);
  EXPECT_EQ(tr_tensor_copy_data(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

TEST(TorchrktRandom, VersionLooksLikeSemver) {
  const std::string v = tr_version();
  EXPECT_FALSE(v.empty());
  EXPECT_NE(v.find('.'), std::string::npos) << "got: " << v;
}

TEST(TorchrktRandom, ShapeAndNumel) {
  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  tr_tensor* t = make_randn({2, 2});

  int64_t numel = 0;
  int64_t ndim = 0;
  ASSERT_EQ(tr_tensor_numel(t, &numel), 0) << tr_last_error();
  ASSERT_EQ(tr_tensor_ndim(t, &ndim), 0) << tr_last_error();
  EXPECT_EQ(numel, 4);
  EXPECT_EQ(ndim, 2);

  int64_t dims[2] = {0, 0};
  int64_t got_ndim = 0;
  ASSERT_EQ(tr_tensor_shape(t, 2, dims, &got_ndim), 0) << tr_last_error();
  EXPECT_EQ(got_ndim, 2);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[1], 2);

  tr_tensor_free(t);
}

TEST(TorchrktRandom, ShapeProbeReportsRequiredNdim) {
  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  tr_tensor* t = make_randn({2, 2});
  int64_t got_ndim = 0;
  EXPECT_EQ(tr_tensor_shape(t, 0, nullptr, &got_ndim), 2);
  EXPECT_EQ(got_ndim, 2);
  tr_tensor_free(t);
}

// The bedrock guarantee we rely on for PyTorch parity: the same seed yields the
// same draws. (Bit-exact agreement with the Python torch lives in the Racket
// python-cross-test.)
TEST(TorchrktRandom, SeedIsDeterministic) {
  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  tr_tensor* a = make_randn({2, 2});
  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  tr_tensor* b = make_randn({2, 2});

  const std::vector<float> da = data_of(a);
  const std::vector<float> db = data_of(b);
  ASSERT_EQ(da.size(), 4u);
  ASSERT_EQ(db.size(), 4u);
  for (size_t i = 0; i < da.size(); ++i) {
    EXPECT_FLOAT_EQ(da[i], db[i]) << "mismatch at " << i;
  }

  tr_tensor_free(a);
  tr_tensor_free(b);
}

TEST(TorchrktRandom, DifferentSeedsDiffer) {
  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  tr_tensor* a = make_randn({2, 2});
  ASSERT_EQ(tr_manual_seed(1), 0) << tr_last_error();
  tr_tensor* b = make_randn({2, 2});

  const std::vector<float> da = data_of(a);
  const std::vector<float> db = data_of(b);
  EXPECT_NE(da, db);

  tr_tensor_free(a);
  tr_tensor_free(b);
}

}  // namespace
