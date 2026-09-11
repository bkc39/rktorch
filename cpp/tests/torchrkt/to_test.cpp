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

Handle from_values(const std::vector<float>& values,
                   const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

std::vector<float> cpu_data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, 0, nullptr, &numel), 2) << tr_last_error();
  std::vector<float> out(numel);
  EXPECT_EQ(tr_tensor_copy_data(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

tr_dtype dtype_of(const tr_tensor* t) {
  tr_dtype out = TR_DTYPE_KEEP;
  EXPECT_EQ(tr_tensor_dtype(t, &out), 0) << tr_last_error();
  return out;
}

tr_device_type device_type_of(const tr_tensor* t) {
  tr_device_type type = TR_DEVICE_KEEP;
  int64_t index = -1;
  EXPECT_EQ(tr_tensor_device(t, &type, &index), 0) << tr_last_error();
  return type;
}

// accumulates d/dx sum(x*x) = 2x into x's grad
void backward_sum_of_squares(tr_tensor* x) {
  const Handle sq(tr_mul(x, x));
  Handle y(tr_sum(sq.t));
  ASSERT_EQ(tr_tensor_backward(y.t), 0) << tr_last_error();
}

void expect_grad(const tr_tensor* x, const std::vector<float>& values) {
  const Handle g(tr_tensor_grad(x));
  EXPECT_EQ(cpu_data_of(g.t), values);
}

TEST(TorchrktTo, KeepBothAliasesTheSource) {
  const Handle src = from_values({1.0F, 2.0F, 3.0F}, {3});
  const Handle same(tr_tensor_to(src.t, TR_DEVICE_KEEP, 0, TR_DTYPE_KEEP));
  const Handle ones = from_values({1.0F, 1.0F, 1.0F}, {3});
  ASSERT_EQ(tr_tensor_sub_(same.t, ones.t, 1.0), 0) << tr_last_error();
  EXPECT_EQ(cpu_data_of(src.t), (std::vector<float>{0.0F, 1.0F, 2.0F}));
}

TEST(TorchrktTo, DtypeOnlyCasts) {
  const Handle src = from_values({1.0F, 2.0F, 3.0F}, {3});
  const Handle wide(tr_tensor_to(src.t, TR_DEVICE_KEEP, 0, TR_DTYPE_FLOAT64));
  EXPECT_EQ(dtype_of(wide.t), TR_DTYPE_FLOAT64);
  EXPECT_EQ(dtype_of(src.t), TR_DTYPE_FLOAT32);
  EXPECT_EQ(device_type_of(wide.t), TR_DEVICE_CPU);
  EXPECT_EQ(cpu_data_of(wide.t), (std::vector<float>{1.0F, 2.0F, 3.0F}));
}

TEST(TorchrktTo, RejectsNullAndUnknownEnums) {
  const Handle src = from_values({1.0F}, {1});
  EXPECT_EQ(tr_tensor_to(nullptr, TR_DEVICE_CPU, 0, TR_DTYPE_KEEP), nullptr);
  EXPECT_EQ(
      tr_tensor_to(src.t, static_cast<tr_device_type>(7), 0, TR_DTYPE_KEEP),
      nullptr);
  EXPECT_NE(strstr(tr_last_error(), "unknown tr_device_type"), nullptr)
      << tr_last_error();
  EXPECT_EQ(tr_tensor_to(src.t, TR_DEVICE_KEEP, 0, static_cast<tr_dtype>(9)),
            nullptr);
  EXPECT_NE(strstr(tr_last_error(), "unknown tr_dtype"), nullptr)
      << tr_last_error();
  EXPECT_EQ(tr_tensor_to_(nullptr, TR_DEVICE_CPU, 0, TR_DTYPE_KEEP), 1);
  EXPECT_EQ(tr_tensor_to_(src.t, TR_DEVICE_KEEP, 0, static_cast<tr_dtype>(9)),
            1);
  EXPECT_EQ(dtype_of(src.t), TR_DTYPE_FLOAT32);
}

TEST(TorchrktTo, InplaceDtypeKeepsHandleLeafAndGrad) {
  Handle x = from_values({1.0F, 2.0F, 3.0F}, {3});
  tr_tensor* const handle = x.t;
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  backward_sum_of_squares(x.t);

  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_KEEP, 0, TR_DTYPE_FLOAT64), 0)
      << tr_last_error();
  EXPECT_EQ(x.t, handle);
  EXPECT_EQ(dtype_of(x.t), TR_DTYPE_FLOAT64);
  EXPECT_EQ(cpu_data_of(x.t), (std::vector<float>{1.0F, 2.0F, 3.0F}));

  int requires_grad = 0;
  ASSERT_EQ(tr_tensor_requires_grad(x.t, &requires_grad), 0) << tr_last_error();
  EXPECT_EQ(requires_grad, 1);
  int has_grad = 0;
  ASSERT_EQ(tr_tensor_has_grad(x.t, &has_grad), 0) << tr_last_error();
  EXPECT_EQ(has_grad, 1);
  {
    const Handle g(tr_tensor_grad(x.t));
    EXPECT_EQ(dtype_of(g.t), TR_DTYPE_FLOAT64);
  }
  expect_grad(x.t, {2.0F, 4.0F, 6.0F});

  // still a leaf: a fresh backward accumulates into the same .grad
  backward_sum_of_squares(x.t);
  expect_grad(x.t, {4.0F, 8.0F, 12.0F});

  // a no-op when nothing changes; casting back restores float32 for both
  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_CPU, 0, TR_DTYPE_FLOAT64), 0)
      << tr_last_error();
  EXPECT_EQ(dtype_of(x.t), TR_DTYPE_FLOAT64);
  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_KEEP, 0, TR_DTYPE_FLOAT32), 0)
      << tr_last_error();
  EXPECT_EQ(dtype_of(x.t), TR_DTYPE_FLOAT32);
  const Handle g(tr_tensor_grad(x.t));
  EXPECT_EQ(dtype_of(g.t), TR_DTYPE_FLOAT32);
}

TEST(TorchrktTo, CudaInplaceRoundTrip) {
  if (tr_cuda_is_available() == 0) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  Handle x = from_values({1.0F, 2.0F, 3.0F}, {3});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  backward_sum_of_squares(x.t);

  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_CUDA, 0, TR_DTYPE_KEEP), 0)
      << tr_last_error();
  EXPECT_EQ(device_type_of(x.t), TR_DEVICE_CUDA);
  EXPECT_EQ(cpu_data_of(x.t), (std::vector<float>{1.0F, 2.0F, 3.0F}));
  {
    const Handle g(tr_tensor_grad(x.t));
    EXPECT_EQ(device_type_of(g.t), TR_DEVICE_CUDA);
  }
  expect_grad(x.t, {2.0F, 4.0F, 6.0F});
  backward_sum_of_squares(x.t);

  const Handle both(tr_tensor_to(x.t, TR_DEVICE_CPU, 0, TR_DTYPE_FLOAT64));
  EXPECT_EQ(device_type_of(both.t), TR_DEVICE_CPU);
  EXPECT_EQ(dtype_of(both.t), TR_DTYPE_FLOAT64);

  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_CPU, 0, TR_DTYPE_KEEP), 0)
      << tr_last_error();
  EXPECT_EQ(device_type_of(x.t), TR_DEVICE_CPU);
  EXPECT_EQ(cpu_data_of(x.t), (std::vector<float>{1.0F, 2.0F, 3.0F}));
  const Handle g(tr_tensor_grad(x.t));
  EXPECT_EQ(device_type_of(g.t), TR_DEVICE_CPU);
  expect_grad(x.t, {4.0F, 8.0F, 12.0F});
}

TEST(TorchrktTo, MpsInplaceRoundTrip) {
  if (tr_mps_is_available() == 0) {
    GTEST_SKIP() << "no MPS device visible";
  }
  Handle x = from_values({1.0F, 2.0F, 3.0F}, {3});
  ASSERT_EQ(tr_tensor_requires_grad_(x.t, 1), 0) << tr_last_error();
  backward_sum_of_squares(x.t);

  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_MPS, 0, TR_DTYPE_KEEP), 0)
      << tr_last_error();
  EXPECT_EQ(device_type_of(x.t), TR_DEVICE_MPS);
  EXPECT_EQ(cpu_data_of(x.t), (std::vector<float>{1.0F, 2.0F, 3.0F}));
  backward_sum_of_squares(x.t);

  ASSERT_EQ(tr_tensor_to_(x.t, TR_DEVICE_CPU, 0, TR_DTYPE_KEEP), 0)
      << tr_last_error();
  EXPECT_EQ(device_type_of(x.t), TR_DEVICE_CPU);
  const Handle g(tr_tensor_grad(x.t));
  EXPECT_EQ(device_type_of(g.t), TR_DEVICE_CPU);
  expect_grad(x.t, {4.0F, 8.0F, 12.0F});
}

TEST(TorchrktCreationOn, PlacesAndTypesAtConstruction) {
  const std::vector<int64_t> dims = {2, 3};
  const Handle z(tr_zeros_on(dims.data(), 2, TR_DEVICE_CPU, 0, TR_DTYPE_INT64));
  EXPECT_EQ(dtype_of(z.t), TR_DTYPE_INT64);
  EXPECT_EQ(device_type_of(z.t), TR_DEVICE_CPU);
  EXPECT_EQ(cpu_data_of(z.t), std::vector<float>(6, 0.0F));
  const Handle o(tr_ones_on(dims.data(), 2, TR_DEVICE_KEEP, 0, TR_DTYPE_KEEP));
  EXPECT_EQ(dtype_of(o.t), TR_DTYPE_FLOAT32);
  EXPECT_EQ(cpu_data_of(o.t), std::vector<float>(6, 1.0F));
  const Handle f(
      tr_full_on(dims.data(), 2, 7.0, TR_DEVICE_KEEP, 0, TR_DTYPE_FLOAT64));
  EXPECT_EQ(dtype_of(f.t), TR_DTYPE_FLOAT64);
  EXPECT_EQ(cpu_data_of(f.t), std::vector<float>(6, 7.0F));
  EXPECT_EQ(tr_zeros_on(nullptr, 2, TR_DEVICE_CPU, 0, TR_DTYPE_KEEP), nullptr);
  EXPECT_EQ(tr_zeros_on(dims.data(), 2, TR_DEVICE_KEEP, 0,
                        static_cast<tr_dtype>(9)),
            nullptr);
}

TEST(TorchrktCreationOn, CudaPlacement) {
  if (tr_cuda_is_available() == 0) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  const std::vector<int64_t> dims = {4};
  const Handle z(tr_zeros_on(dims.data(), 1, TR_DEVICE_CUDA, 0, TR_DTYPE_KEEP));
  EXPECT_EQ(device_type_of(z.t), TR_DEVICE_CUDA);
  EXPECT_EQ(cpu_data_of(z.t), std::vector<float>(4, 0.0F));
}

}  // namespace
