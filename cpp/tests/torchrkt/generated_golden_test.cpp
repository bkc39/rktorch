#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "torchrkt/c_api.h"

// Golden-equivalence proof for the codegen pipeline (#2): the generated
// tr_gen_* linalg ops must be bit-identical to the hand-written tr_* ops
// they shadow. The hand-written family stays authoritative; this test is
// what licenses trusting the generator for ops that have no hand-written
// twin.

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

void expect_bit_identical(const tr_tensor* expected, const tr_tensor* actual) {
  EXPECT_EQ(shape_of(expected), shape_of(actual));
  EXPECT_EQ(data_of(expected), data_of(actual));
}

TEST(GeneratedGolden, MatmulMatchesHandWritten) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3});
  const Handle b = make({0.5F, -1.0F, 2.0F, 0.25F, -3.0F, 1.5F}, {3, 2});
  const Handle expected(tr_matmul(a.t, b.t));
  const Handle actual(tr_gen_matmul(a.t, b.t));
  expect_bit_identical(expected.t, actual.t);
}

TEST(GeneratedGolden, MmMatchesHandWritten) {
  const Handle a = make({1.0F, -2.0F, 3.5F, 4.0F}, {2, 2});
  const Handle b = make({0.5F, 1.0F, -1.5F, 2.0F}, {2, 2});
  const Handle expected(tr_mm(a.t, b.t));
  const Handle actual(tr_gen_mm(a.t, b.t));
  expect_bit_identical(expected.t, actual.t);
}

TEST(GeneratedGolden, MvMatchesHandWritten) {
  const Handle m = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3});
  const Handle v = make({0.5F, -1.0F, 2.0F}, {3});
  const Handle expected(tr_mv(m.t, v.t));
  const Handle actual(tr_gen_mv(m.t, v.t));
  expect_bit_identical(expected.t, actual.t);
}

TEST(GeneratedGolden, DotMatchesHandWritten) {
  const Handle a = make({1.0F, 2.0F, 3.0F}, {3});
  const Handle b = make({4.0F, -5.0F, 6.0F}, {3});
  const Handle expected(tr_dot(a.t, b.t));
  const Handle actual(tr_gen_dot(a.t, b.t));
  expect_bit_identical(expected.t, actual.t);
}

TEST(GeneratedGolden, ErrorsSurfaceAsNullNotAbort) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const Handle v = make({1.0F, 2.0F, 3.0F}, {3});
  EXPECT_EQ(tr_gen_mm(a.t, v.t), nullptr);
  EXPECT_NE(tr_last_error(), nullptr);
  EXPECT_EQ(tr_gen_matmul(nullptr, a.t), nullptr);
}

}  // namespace
