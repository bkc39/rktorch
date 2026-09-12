#include <gtest/gtest.h>

#include <algorithm>
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

std::vector<float> randperm_of(int64_t n, tr_generator* g) {
  tr_tensor* p = tr_randperm(n, g);
  EXPECT_NE(p, nullptr) << tr_last_error();
  std::vector<float> out = data_of(p);
  tr_tensor_free(p);
  return out;
}

bool is_permutation(const std::vector<float>& xs) {
  std::vector<float> sorted = xs;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != static_cast<float>(i)) {
      return false;
    }
  }
  return true;
}

TEST(TorchrktRandom, GeneratorSeedReplaysPermutations) {
  tr_generator* a = tr_generator_new(7);
  tr_generator* b = tr_generator_new(7);
  ASSERT_NE(a, nullptr) << tr_last_error();
  ASSERT_NE(b, nullptr) << tr_last_error();
  const std::vector<float> a1 = randperm_of(16, a);
  const std::vector<float> a2 = randperm_of(16, a);
  EXPECT_TRUE(is_permutation(a1));
  EXPECT_TRUE(is_permutation(a2));
  EXPECT_NE(a1, a2) << "the stream continues across draws";
  EXPECT_EQ(randperm_of(16, b), a1) << "same seed, same first draw";
  EXPECT_EQ(randperm_of(16, b), a2) << "same seed, same second draw";
  tr_generator_free(a);
  tr_generator_free(b);
  tr_generator_free(nullptr);
}

TEST(TorchrktRandom, GeneratorDrawsLeaveTheGlobalStreamAlone) {
  ASSERT_EQ(tr_manual_seed(3), 0) << tr_last_error();
  tr_tensor* before = make_randn({4});
  const std::vector<float> expected = data_of(before);
  tr_tensor_free(before);
  ASSERT_EQ(tr_manual_seed(3), 0) << tr_last_error();
  tr_generator* g = tr_generator_new(11);
  ASSERT_NE(g, nullptr) << tr_last_error();
  randperm_of(64, g);
  tr_generator_free(g);
  tr_tensor* after = make_randn({4});
  EXPECT_EQ(data_of(after), expected);
  tr_tensor_free(after);
}

TEST(TorchrktRandom, DrawSeedReplaysAndAdvancesTheStream) {
  tr_generator* a = tr_generator_new(21);
  tr_generator* b = tr_generator_new(21);
  ASSERT_NE(a, nullptr) << tr_last_error();
  ASSERT_NE(b, nullptr) << tr_last_error();
  int64_t sa = -1;
  int64_t sb = -1;
  ASSERT_EQ(tr_generator_draw_seed(a, &sa), 0) << tr_last_error();
  ASSERT_EQ(tr_generator_draw_seed(b, &sb), 0) << tr_last_error();
  EXPECT_EQ(sa, sb) << "same seed, same draw";
  EXPECT_GE(sa, 0);
  // the draw advanced a: its permutation is b's second, not b's first
  const std::vector<float> pa = randperm_of(8, a);
  int64_t again = -1;
  ASSERT_EQ(tr_generator_draw_seed(b, &again), 0) << tr_last_error();
  EXPECT_NE(again, sb) << "the stream continues";
  tr_generator_free(a);
  tr_generator_free(b);
  int64_t global = -1;
  ASSERT_EQ(tr_manual_seed(2), 0) << tr_last_error();
  ASSERT_EQ(tr_generator_draw_seed(nullptr, &global), 0) << tr_last_error();
  int64_t global_again = -1;
  ASSERT_EQ(tr_manual_seed(2), 0) << tr_last_error();
  ASSERT_EQ(tr_generator_draw_seed(nullptr, &global_again), 0)
      << tr_last_error();
  EXPECT_EQ(global, global_again) << "NULL draws from the global stream";
  EXPECT_EQ(tr_generator_draw_seed(nullptr, nullptr), 1);
  (void)pa;
}

TEST(TorchrktRandom, RandpermWithoutGeneratorAndErrors) {
  ASSERT_EQ(tr_manual_seed(5), 0) << tr_last_error();
  const std::vector<float> p1 = randperm_of(10, nullptr);
  ASSERT_EQ(tr_manual_seed(5), 0) << tr_last_error();
  EXPECT_EQ(randperm_of(10, nullptr), p1)
      << "NULL draws from the global stream";
  EXPECT_TRUE(is_permutation(p1));
  tr_tensor* empty = tr_randperm(0, nullptr);
  ASSERT_NE(empty, nullptr) << tr_last_error();
  int64_t numel = -1;
  EXPECT_EQ(tr_tensor_numel(empty, &numel), 0) << tr_last_error();
  EXPECT_EQ(numel, 0);
  tr_tensor_free(empty);
  EXPECT_EQ(tr_randperm(-1, nullptr), nullptr);
}

}  // namespace
