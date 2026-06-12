#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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

TEST(GeneratedGolden, ReshapeMatchesHandWritten) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3});
  const std::vector<int64_t> dims = {3, 2};
  const Handle expected(tr_reshape(a.t, dims.data(), 2));
  const Handle actual(tr_gen_reshape(a.t, dims.data(), 2));
  expect_bit_identical(expected.t, actual.t);
}

// A -1 *value* in the shape array is ATen's inferred dimension and must
// pass the guard — only a negative *count* (shape_len) is rejected.
TEST(GeneratedGolden, ReshapeInferredDimMatchesHandWritten) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3});
  const std::vector<int64_t> dims = {-1, 2};
  const Handle expected(tr_reshape(a.t, dims.data(), 2));
  const Handle actual(tr_gen_reshape(a.t, dims.data(), 2));
  EXPECT_EQ(shape_of(actual.t), (std::vector<int64_t>{3, 2}));
  expect_bit_identical(expected.t, actual.t);
}

TEST(GeneratedGolden, CatMatchesHandWritten) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const Handle b = make({5.0F, 6.0F, 7.0F, 8.0F}, {2, 2});
  const tr_tensor* parts[] = {a.t, b.t};
  const Handle expected(tr_cat(parts, 2, 0));
  const Handle actual(tr_gen_cat(parts, 2, 0));
  expect_bit_identical(expected.t, actual.t);
}

// tr_last_error is a sticky thread-local ("only valid after a failure"
// contract — success does not clear it), so each check below asserts the
// message is attributed to the op that just failed rather than merely
// non-null, which would pass on a stale string from an earlier case.
void expect_error_from(const char* who) {
  const char* message = tr_last_error();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(std::strstr(message, who), nullptr) << message;
}

TEST(GeneratedGolden, ErrorsSurfaceAsNullNotAbort) {
  const Handle a = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  const Handle v = make({1.0F, 2.0F, 3.0F}, {3});
  EXPECT_EQ(tr_gen_mm(a.t, v.t), nullptr);
  expect_error_from("tr_gen_mm");

  // Every generated binary op carries the same null-arg guard.
  struct BinaryCase {
    const char* name;
    tr_tensor* (*fn)(const tr_tensor*, const tr_tensor*);
  };
  const BinaryCase binary_cases[] = {
      {"tr_gen_matmul", tr_gen_matmul},
      {"tr_gen_mm", tr_gen_mm},
      {"tr_gen_mv", tr_gen_mv},
      {"tr_gen_dot", tr_gen_dot},
  };
  for (const auto& c : binary_cases) {
    EXPECT_EQ(c.fn(nullptr, a.t), nullptr) << c.name;
    expect_error_from(c.name);
    EXPECT_EQ(c.fn(a.t, nullptr), nullptr) << c.name;
    expect_error_from(c.name);
  }

  // The TensorList path converts a null element into the error contract
  // (via the generated throw), never a crash.
  const tr_tensor* holey[] = {a.t, nullptr};
  EXPECT_EQ(tr_gen_cat(holey, 2, 0), nullptr);
  expect_error_from("tr_gen_cat");
  // A zero-length TensorList passes the pointer guard; ATen's own throw
  // (cat of an empty list) surfaces through alloc_result as NULL.
  EXPECT_EQ(tr_gen_cat(holey, 0, 0), nullptr);
  expect_error_from("tr_gen_cat");
  // Negative lengths are rejected before any pointer arithmetic.
  EXPECT_EQ(tr_gen_reshape(a.t, nullptr, -1), nullptr);
  expect_error_from("tr_gen_reshape");
}

}  // namespace
