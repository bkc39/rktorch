#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "torchrkt/c_api.h"

namespace {

std::vector<std::uint8_t> bytes_of(const tr_tensor* t) {
  std::uint64_t n = 0;
  EXPECT_EQ(tr_tensor_copy_bytes(t, 0, nullptr, &n), 2) << tr_last_error();
  std::vector<std::uint8_t> out(n);
  EXPECT_EQ(tr_tensor_copy_bytes(t, n, out.data(), &n), 0) << tr_last_error();
  return out;
}

std::vector<std::uint8_t> permutation_of(tr_generator* g) {
  tr_tensor* p = tr_randperm(8, g);
  EXPECT_NE(p, nullptr) << tr_last_error();
  std::vector<std::uint8_t> out = bytes_of(p);
  tr_tensor_free(p);
  return out;
}

TEST(TorchrktRngState, GeneratorStateIsBytesAndReplays) {
  tr_generator* g = tr_generator_new(5);
  tr_tensor* s = tr_generator_state(g);
  ASSERT_NE(s, nullptr) << tr_last_error();
  tr_dtype dtype = TR_DTYPE_KEEP;
  int64_t numel = 0;
  ASSERT_EQ(tr_tensor_dtype(s, &dtype), 0) << tr_last_error();
  ASSERT_EQ(tr_tensor_numel(s, &numel), 0) << tr_last_error();
  EXPECT_EQ(dtype, TR_DTYPE_UINT8);
  EXPECT_EQ(numel, 5056);
  const std::vector<std::uint8_t> first = permutation_of(g);
  ASSERT_EQ(tr_generator_set_state(g, s), 0) << tr_last_error();
  EXPECT_EQ(permutation_of(g), first);
  tr_tensor_free(s);
  tr_generator_free(g);
}

TEST(TorchrktRngState, ManualSeedSetsTheDefaultStateASeededGeneratorHas) {
  ASSERT_EQ(tr_manual_seed(9), 0) << tr_last_error();
  tr_generator* g = tr_generator_new(9);
  tr_tensor* d = tr_default_generator_state(TR_DEVICE_CPU, 0);
  tr_tensor* s = tr_generator_state(g);
  ASSERT_NE(d, nullptr) << tr_last_error();
  ASSERT_NE(s, nullptr) << tr_last_error();
  EXPECT_EQ(bytes_of(d), bytes_of(s));
  tr_tensor_free(d);
  tr_tensor_free(s);
  tr_generator_free(g);
}

TEST(TorchrktRngState, StateCallsFailCleanly) {
  tr_generator* g = tr_generator_new(1);
  tr_tensor* s = tr_generator_state(g);
  const std::vector<std::uint8_t> zeros(16, 0);
  const int64_t sixteen = 16;
  tr_tensor* cuda_shaped =
      tr_from_bytes(zeros.data(), zeros.size(), &sixteen, 1, TR_DTYPE_UINT8);
  ASSERT_NE(s, nullptr) << tr_last_error();
  ASSERT_NE(cuda_shaped, nullptr) << tr_last_error();
  const auto failed = [](bool refused) {
    EXPECT_TRUE(refused);
    EXPECT_STRNE(tr_last_error(), "");
  };
  failed(tr_generator_state(nullptr) == nullptr);
  failed(tr_generator_set_state(nullptr, s) == 1);
  failed(tr_generator_set_state(g, nullptr) == 1);
  failed(tr_default_generator_set_state(TR_DEVICE_CPU, 0, nullptr) == 1);
  failed(tr_default_generator_state(TR_DEVICE_KEEP, 0) == nullptr);
  failed(tr_default_generator_set_state(TR_DEVICE_KEEP, 0, s) == 1);
  failed(tr_default_generator_state(TR_DEVICE_CUDA, 99) == nullptr);
  failed(tr_generator_set_state(g, cuda_shaped) == 1);
  tr_tensor_free(cuda_shaped);
  tr_tensor_free(s);
  tr_generator_free(g);
}

TEST(TorchrktRngState, StateAtAStorageOffsetReplays) {
  tr_generator* g = tr_generator_new(3);
  tr_tensor* s = tr_generator_state(g);
  ASSERT_NE(s, nullptr) << tr_last_error();
  std::vector<std::uint8_t> padded(4, 0);
  const std::vector<std::uint8_t> state = bytes_of(s);
  padded.insert(padded.end(), state.begin(), state.end());
  const auto n = static_cast<int64_t>(padded.size());
  tr_tensor* whole =
      tr_from_bytes(padded.data(), padded.size(), &n, 1, TR_DTYPE_UINT8);
  ASSERT_NE(whole, nullptr) << tr_last_error();
  tr_tensor* view = tr_gen_narrow(whole, 0, 4, n - 4);
  ASSERT_NE(view, nullptr) << tr_last_error();
  const std::vector<std::uint8_t> first = permutation_of(g);
  ASSERT_EQ(tr_generator_set_state(g, view), 0) << tr_last_error();
  EXPECT_EQ(permutation_of(g), first);
  for (tr_tensor* t : {s, whole, view}) {
    tr_tensor_free(t);
  }
  tr_generator_free(g);
}

}  // namespace
