#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "torchrkt/c_api.h"

// Goldens for representative ops per family (creation, shape, elementwise,
// reduce, linalg, marshalling). Exhaustive value parity with PyTorch lives in
// the Racket python-cross-test; these pin the C contract.

namespace {

// RAII so a failed EXPECT can't leak handles past the test body.
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

// Contract pin for the documented NULL no-op. The throwing-release
// behavior is pinned separately by finalizer_death_test.cpp (it
// terminates regardless of any C++ catch; the live guarantee is the
// Racket-side deallocator wrap in raw/memory.rkt).
TEST(TorchrktOps, TensorFreeNullIsSafe) {
  tr_tensor_free(nullptr);
}

TEST(TorchrktOps, NbytesTracksDtypeWidth) {
  const Handle t = make({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  int64_t nbytes = 0;
  ASSERT_EQ(tr_tensor_nbytes(t.t, &nbytes), 0) << tr_last_error();
  EXPECT_EQ(nbytes, 16);  // 4 x float32
  const Handle i64 = Handle(tr_tensor_to_dtype(t.t, TR_DTYPE_INT64));
  int64_t nbytes64 = 0;
  ASSERT_EQ(tr_tensor_nbytes(i64.t, &nbytes64), 0) << tr_last_error();
  EXPECT_EQ(nbytes64, 32);  // 4 x int64
  EXPECT_EQ(tr_tensor_nbytes(nullptr, &nbytes), 1);
}

TEST(TorchrktOps, NbytesReportsViewExtentNotStorage) {
  // The #37 ledger's documented approximation: a view charges what it
  // ADDRESSES, not the (possibly larger, shared) storage behind it. A
  // narrow over half the rows must report half the bytes even though its
  // storage is the full tensor's.
  const Handle t =
      make({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}, {4, 2});
  const Handle half = Handle(tr_gen_narrow(t.t, 0, 0, 2));  // rows [0,2)
  int64_t full_bytes = 0;
  int64_t view_bytes = 0;
  ASSERT_EQ(tr_tensor_nbytes(t.t, &full_bytes), 0) << tr_last_error();
  ASSERT_EQ(tr_tensor_nbytes(half.t, &view_bytes), 0) << tr_last_error();
  EXPECT_EQ(full_bytes, 32);  // 8 x float32
  EXPECT_EQ(view_bytes, 16);  // the view's 4 elements, not storage's 8
}

TEST(TorchrktOps, FromDataRoundTrips) {
  const std::vector<float> values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const Handle t = make(values, {2, 3});
  EXPECT_EQ(shape_of(t.t), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(data_of(t.t), values);
}

TEST(TorchrktOps, FromDataI64RoundTripsExactly) {
  // The #44 inference path: int64 in, int64 out — no float32 transit.
  // 2^53+1 is NOT representable as a double, so a float32/64 transit
  // would corrupt it; exact round-trip proves the path is integral.
  const std::vector<int64_t> values = {1, -2, (int64_t{1} << 53) + 1};
  const std::vector<int64_t> dims = {3};
  const Handle t(
      tr_from_data_i64(values.data(), values.size(), dims.data(), 1));
  ASSERT_NE(t.t, nullptr) << tr_last_error();
  tr_dtype dt = TR_DTYPE_FLOAT32;
  EXPECT_EQ(tr_tensor_dtype(t.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_INT64);
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data_i64(t.t, 0, nullptr, &numel), 2);
  std::vector<int64_t> out(numel);
  EXPECT_EQ(tr_tensor_copy_data_i64(t.t, numel, out.data(), &numel), 0)
      << tr_last_error();
  EXPECT_EQ(out, values);
}

TEST(TorchrktOps, FromDataI64RejectsBadShapes) {
  // the float sibling's rejection battery, on the int64 path
  const std::vector<int64_t> values = {1, 2, 3};
  const std::vector<int64_t> dims = {2, 2};
  EXPECT_EQ(tr_from_data_i64(values.data(), values.size(), dims.data(), 2),
            nullptr);
  EXPECT_STRNE(tr_last_error(), "");
  const std::vector<int64_t> overflow = {2, 9223372036854775807LL, 2};
  EXPECT_EQ(tr_from_data_i64(values.data(), values.size(), overflow.data(), 3),
            nullptr);
  const std::vector<int64_t> negative = {-1, 2};
  EXPECT_EQ(tr_from_data_i64(values.data(), values.size(), negative.data(), 2),
            nullptr);
  EXPECT_EQ(tr_from_data_i64(nullptr, 3, dims.data(), 2), nullptr);
}

TEST(TorchrktOps, DtypeGetterCoversTheEnum) {
  const std::vector<float> values = {1.0F};
  const std::vector<int64_t> dims = {1};
  const Handle f32(tr_from_data(values.data(), values.size(), dims.data(), 1));
  tr_dtype dt = TR_DTYPE_INT64;
  EXPECT_EQ(tr_tensor_dtype(f32.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_FLOAT32);
  const Handle f64(tr_tensor_to_dtype(f32.t, TR_DTYPE_FLOAT64));
  EXPECT_EQ(tr_tensor_dtype(f64.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_FLOAT64);
  EXPECT_EQ(tr_tensor_dtype(nullptr, &dt), 1);
  EXPECT_STRNE(tr_last_error(), "");
}

TEST(TorchrktOps, DtypeGetterReportsBoolMasks) {
  const std::vector<float> values = {1.0F, 2.0F};
  const std::vector<int64_t> dims = {2};
  const Handle t(tr_from_data(values.data(), values.size(), dims.data(), 1));
  const Handle mask(tr_gen_eq_scalar(t.t, 1.0));
  tr_dtype dt = TR_DTYPE_FLOAT32;
  EXPECT_EQ(tr_tensor_dtype(mask.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_BOOL);
  // and to_dtype CASTS to bool (nonzero -> true), not just identity
  const Handle cast(tr_tensor_to_dtype(t.t, TR_DTYPE_BOOL));
  EXPECT_EQ(tr_tensor_dtype(cast.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_BOOL);
}

TEST(TorchrktOps, FromDataI64OnPlacesOnExplicitCpu) {
  // the device-move leg of the int64 ingestion, exercised at the C
  // boundary (its float sibling has the same coverage in device_test)
  const std::vector<int64_t> values = {1, -2, (int64_t{1} << 53) + 1};
  const std::vector<int64_t> dims = {3};
  const Handle t(tr_from_data_i64_on(values.data(), values.size(), dims.data(),
                                     1, TR_DEVICE_CPU, 0));
  ASSERT_NE(t.t, nullptr) << tr_last_error();
  tr_dtype dt = TR_DTYPE_FLOAT32;
  EXPECT_EQ(tr_tensor_dtype(t.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_INT64);
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data_i64(t.t, 0, nullptr, &numel), 2);
  std::vector<int64_t> out(numel);
  EXPECT_EQ(tr_tensor_copy_data_i64(t.t, numel, out.data(), &numel), 0)
      << tr_last_error();
  EXPECT_EQ(out, values);
  // invalid device rejection mirrors the float sibling
  EXPECT_EQ(tr_from_data_i64_on(values.data(), values.size(), dims.data(), 1,
                                static_cast<tr_device_type>(99), 0),
            nullptr);
}

TEST(TorchrktOps, FromDataAcceptsNullForEmptyTensors) {
  // numel==0 constructs with a NULL data pointer (an empty Racket
  // vector marshals as NULL); nonzero numel still rejects NULL.
  const std::vector<int64_t> rank1 = {0};
  const Handle e1(tr_from_data(nullptr, 0, rank1.data(), 1));
  ASSERT_NE(e1.t, nullptr) << tr_last_error();
  std::uint64_t numel = 99;
  EXPECT_EQ(tr_tensor_copy_data(e1.t, 0, nullptr, &numel), 0);
  EXPECT_EQ(numel, 0U);
  const std::vector<int64_t> rank2 = {2, 0};
  const Handle e2(tr_from_data_i64(nullptr, 0, rank2.data(), 2));
  ASSERT_NE(e2.t, nullptr) << tr_last_error();
  tr_dtype dt = TR_DTYPE_FLOAT32;
  EXPECT_EQ(tr_tensor_dtype(e2.t, &dt), 0) << tr_last_error();
  EXPECT_EQ(dt, TR_DTYPE_INT64);
  EXPECT_EQ(tr_from_data(nullptr, 3, rank1.data(), 1), nullptr);
  EXPECT_STRNE(tr_last_error(), "");
}

TEST(TorchrktOps, CopyDataF64PreservesDoublePrecision) {
  // 2^24+1 is unrepresentable in float32; the f64 path keeps it exact
  const std::vector<int64_t> ival = {16777217};
  const std::vector<int64_t> dims = {1};
  const Handle i(tr_from_data_i64(ival.data(), 1, dims.data(), 1));
  const Handle d(tr_tensor_to_dtype(i.t, TR_DTYPE_FLOAT64));
  std::uint64_t numel = 0;
  double out = 0.0;
  EXPECT_EQ(tr_tensor_copy_data_f64(d.t, 1, &out, &numel), 0)
      << tr_last_error();
  EXPECT_EQ(out, 16777217.0);
  float f32out = 0.0F;
  EXPECT_EQ(tr_tensor_copy_data(d.t, 1, &f32out, &numel), 0);
  EXPECT_EQ(f32out, 16777216.0F);
}

TEST(TorchrktOps, FromDataRejectsNumelMismatch) {
  const std::vector<float> values = {1.0F, 2.0F, 3.0F};
  const std::vector<int64_t> dims = {2, 2};
  EXPECT_EQ(tr_from_data(values.data(), values.size(), dims.data(), 2),
            nullptr);
}

TEST(TorchrktOps, FromDataRejectsOverflowingShape) {
  // A dim product that wraps uint64 must not bypass the numel check.
  const std::vector<float> values = {1.0F, 2.0F};
  const std::vector<int64_t> dims = {2, 9223372036854775807LL, 2};
  EXPECT_EQ(tr_from_data(values.data(), values.size(), dims.data(), 3),
            nullptr);
  const std::vector<int64_t> negative = {-1, 2};
  EXPECT_EQ(tr_from_data(values.data(), values.size(), negative.data(), 2),
            nullptr);
}

TEST(TorchrktOps, CpuOomClassifiesAsOomKind) {
  // The portable OOM regression guard leg 1.5's design leans on: CPU
  // allocation failure arrives via the caffe2-style enforce (a plain
  // c10::Error naming DefaultCPUAllocator, NOT c10::OutOfMemoryError),
  // and the classifier must still report kind 1. 2^60 floats = 2^62
  // bytes (4 EiB): beyond the ARCHITECTURAL user address space of every
  // 64-bit platform -- at most 2^56/2^57 virtual bits even with x86-64
  // 5-level paging or ARM64 52-bit VA -- so no allocator can map it
  // regardless of kernel policy or overcommit mode (an architecture
  // bound, not an assumption; no fault-in-pages hazard). nbytes stays
  // below INT64_MAX so ATen's own arithmetic cannot overflow into a
  // different error shape.
  const std::vector<int64_t> dims = {int64_t{1} << 60};
  EXPECT_EQ(tr_zeros(dims.data(), 1), nullptr);
  EXPECT_EQ(tr_last_error_kind(), 1) << tr_last_error();
  EXPECT_STRNE(tr_last_error(), "");
}

TEST(TorchrktOps, GenericErrorResetsKind) {
  // kind and message are recorded together: after an OOM, a subsequent
  // generic failure must report kind 0 again, never a stale 1.
  // (2^60 floats: the beyond-architectural-VA request, see
  // CpuOomClassifiesAsOomKind.)
  const std::vector<int64_t> dims = {int64_t{1} << 60};
  EXPECT_EQ(tr_zeros(dims.data(), 1), nullptr);
  EXPECT_EQ(tr_last_error_kind(), 1) << tr_last_error();
  int64_t out = 0;
  EXPECT_EQ(tr_tensor_numel(nullptr, &out), 1);
  EXPECT_EQ(tr_last_error_kind(), 0) << tr_last_error();
}

TEST(TorchrktOps, CreationGoldens) {
  const std::vector<int64_t> dims = {2, 2};
  const Handle z(tr_zeros(dims.data(), 2));
  EXPECT_EQ(data_of(z.t), (std::vector<float>{0, 0, 0, 0}));

  const Handle o(tr_ones(dims.data(), 2));
  EXPECT_EQ(data_of(o.t), (std::vector<float>{1, 1, 1, 1}));

  const Handle f(tr_full(dims.data(), 2, 7.5));
  EXPECT_EQ(data_of(f.t), (std::vector<float>{7.5F, 7.5F, 7.5F, 7.5F}));

  const Handle a(tr_arange(0.0, 3.0, 1.0));
  EXPECT_EQ(data_of(a.t), (std::vector<float>{0, 1, 2}));

  const Handle e(tr_eye(2, 2));
  EXPECT_EQ(data_of(e.t), (std::vector<float>{1, 0, 0, 1}));
}

TEST(TorchrktOps, ShapeOps) {
  const Handle t = make({1, 2, 3, 4, 5, 6}, {2, 3});

  const std::vector<int64_t> flat = {6};
  const Handle r(tr_reshape(t.t, flat.data(), 1));
  EXPECT_EQ(shape_of(r.t), flat);

  const std::vector<int64_t> three_two = {3, 2};
  const Handle v(tr_view(t.t, three_two.data(), 2));
  EXPECT_EQ(shape_of(v.t), three_two);
  EXPECT_EQ(data_of(v.t), (std::vector<float>{1, 2, 3, 4, 5, 6}));

  const std::vector<int64_t> perm = {1, 0};
  const Handle p(tr_permute(t.t, perm.data(), 2));
  EXPECT_EQ(shape_of(p.t), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(data_of(p.t), (std::vector<float>{1, 4, 2, 5, 3, 6}));

  const Handle tr(tr_transpose(t.t, 0, 1));
  EXPECT_EQ(shape_of(tr.t), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(data_of(tr.t), (std::vector<float>{1, 4, 2, 5, 3, 6}));

  const Handle u(tr_unsqueeze(t.t, 0));
  EXPECT_EQ(shape_of(u.t), (std::vector<int64_t>{1, 2, 3}));
  const Handle s(tr_squeeze(u.t));
  EXPECT_EQ(shape_of(s.t), (std::vector<int64_t>{2, 3}));
  const Handle sd(tr_squeeze_dim(u.t, 0));
  EXPECT_EQ(shape_of(sd.t), (std::vector<int64_t>{2, 3}));

  const tr_tensor* pair[] = {t.t, t.t};
  const Handle c(tr_cat(pair, 2, 0));
  EXPECT_EQ(shape_of(c.t), (std::vector<int64_t>{4, 3}));
  const Handle st(tr_stack(pair, 2, 0));
  EXPECT_EQ(shape_of(st.t), (std::vector<int64_t>{2, 2, 3}));
}

TEST(TorchrktOps, Elementwise) {
  const Handle a = make({1, -2, 3, -4}, {2, 2});
  const Handle b = make({10, 20, 30, 40}, {2, 2});

  const Handle sum(tr_add(a.t, b.t));
  EXPECT_EQ(data_of(sum.t), (std::vector<float>{11, 18, 33, 36}));

  const Handle scaled(tr_mul_scalar(a.t, 2.0));
  EXPECT_EQ(data_of(scaled.t), (std::vector<float>{2, -4, 6, -8}));

  const Handle rect(tr_relu(a.t));
  EXPECT_EQ(data_of(rect.t), (std::vector<float>{1, 0, 3, 0}));

  const Handle squared(tr_pow_scalar(a.t, 2.0));
  EXPECT_EQ(data_of(squared.t), (std::vector<float>{1, 4, 9, 16}));

  // gelu (exact form, approximate='none'): x * Phi(x). gelu(0)=0, and the
  // +-1 values pin the erf-based branch against the tanh approximation.
  const Handle unit = make({0.0F, 1.0F, -1.0F}, {3});
  const Handle smoothed(tr_gelu(unit.t));
  const std::vector<float> g = data_of(smoothed.t);
  EXPECT_FLOAT_EQ(g.at(0), 0.0F);
  EXPECT_NEAR(g.at(1), 0.841345F, 1e-5F);
  EXPECT_NEAR(g.at(2), -0.158655F, 1e-5F);
  EXPECT_EQ(tr_gelu(nullptr), nullptr);
}

TEST(TorchrktOps, ReduceAndItem) {
  const Handle t = make({1, 2, 3, 4}, {2, 2});

  const Handle s(tr_sum(t.t));
  double item = 0.0;
  ASSERT_EQ(tr_tensor_item(s.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 10.0);

  const Handle m(tr_mean(t.t));
  ASSERT_EQ(tr_tensor_item(m.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 2.5);

  const Handle mx(tr_max(t.t));
  ASSERT_EQ(tr_tensor_item(mx.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 4.0);

  const Handle mn(tr_min(t.t));
  ASSERT_EQ(tr_tensor_item(mn.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 1.0);

  const Handle am(tr_argmax_all(t.t));
  ASSERT_EQ(tr_tensor_item(am.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 3.0);

  const Handle sm(tr_softmax(t.t, 1));
  const std::vector<float> probs = data_of(sm.t);
  EXPECT_NEAR(probs[0] + probs[1], 1.0F, 1e-6F);
  EXPECT_NEAR(probs[2] + probs[3], 1.0F, 1e-6F);

  // item on a multi-element tensor must error, not crash.
  EXPECT_EQ(tr_tensor_item(t.t, &item), 1);
}

TEST(TorchrktOps, Linalg) {
  const Handle a = make({1, 2, 3, 4}, {2, 2});
  const Handle b = make({5, 6, 7, 8}, {2, 2});
  const Handle v = make({1, 1}, {2});

  const Handle mm(tr_matmul(a.t, b.t));
  EXPECT_EQ(data_of(mm.t), (std::vector<float>{19, 22, 43, 50}));

  const Handle mv(tr_mv(a.t, v.t));
  EXPECT_EQ(data_of(mv.t), (std::vector<float>{3, 7}));

  const Handle d(tr_dot(v.t, v.t));
  double item = 0.0;
  ASSERT_EQ(tr_tensor_item(d.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 2.0);

  // Shape mismatch surfaces as NULL + message, not an abort.
  const Handle bad = make({1, 2, 3}, {3});
  EXPECT_EQ(tr_mv(a.t, bad.t), nullptr);
}

TEST(TorchrktOps, ToDtypeAndUniform) {
  const Handle t = make({1.5F, 2.5F}, {2});
  const Handle i(tr_tensor_to_dtype(t.t, TR_DTYPE_INT64));
  // Reading an INT64 tensor through data_of is safe by contract:
  // tr_tensor_copy_data converts to CPU float32 contiguous before copying
  // (see c_api/tensor.h), so this is a conversion, not a reinterpret.
  EXPECT_EQ(data_of(i.t), (std::vector<float>{1, 2}));

  ASSERT_EQ(tr_manual_seed(0), 0) << tr_last_error();
  const std::vector<int64_t> dims = {32};
  const Handle u(tr_rand(dims.data(), 1));
  for (const float x : data_of(u.t)) {
    EXPECT_GE(x, 0.0F);
    EXPECT_LT(x, 1.0F);
  }

  Handle w(tr_zeros(dims.data(), 1));
  ASSERT_EQ(tr_tensor_uniform_(w.t, -2.0, -1.0), 0) << tr_last_error();
  for (const float x : data_of(w.t)) {
    EXPECT_GE(x, -2.0F);
    EXPECT_LT(x, -1.0F);
  }
}

}  // namespace
