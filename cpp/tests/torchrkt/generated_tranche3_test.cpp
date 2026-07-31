#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "torchrkt/c_api.h"

// C-boundary goldens for the tranche-3 generated families (#22): one
// correctness case + one error-path case per family. Value parity with
// PyTorch is the Racket python-cross-test's job; these pin the C contract
// the parity battery can't see — null guards, the optional-tensor nullopt
// encoding, and the dtype contracts (int64 indices, bool masks) ATen
// enforces at this boundary.

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

void expect_error_from(const char* who) {
  const char* message = tr_last_error();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(std::strstr(message, who), nullptr) << message;
}

// ---- embedding: int64 row-gather + dtype contract + null guard ---------

TEST(GeneratedTranche3, EmbeddingGathersRows) {
  const Handle weight = make({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
  const Handle indices_f = make({2.0F, 0.0F, 2.0F}, {3});
  const Handle indices(tr_tensor_to_dtype(indices_f.t, TR_DTYPE_INT64));
  const Handle out(tr_gen_embedding(weight.t, indices.t, /*padding_idx=*/-1,
                                    /*scale_grad_by_freq=*/false,
                                    /*sparse=*/false));
  EXPECT_EQ(shape_of(out.t), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(data_of(out.t),
            (std::vector<float>{5.0F, 6.0F, 1.0F, 2.0F, 5.0F, 6.0F}));
  // ATen rejects non-integer indices — the error surfaces as NULL + message,
  // never an abort (the loader's to-dtype 'int64 bridge is load-bearing).
  EXPECT_EQ(tr_gen_embedding(weight.t, indices_f.t, -1, false, false), nullptr);
  expect_error_from("tr_gen_embedding");
  EXPECT_EQ(tr_gen_embedding(nullptr, indices.t, -1, false, false), nullptr);
  expect_error_from("tr_gen_embedding");
}

// ---- layer_norm: affine + bare paths (optional-tensor nullopt) ---------

TEST(GeneratedTranche3, LayerNormAffineAndBare) {
  // Two rows; per-row normalization over the trailing dim of 3. Row {1,2,3}
  // has mean 2 and biased var 2/3, so it normalizes to ±1.2247, 0.
  const Handle x = make({1, 2, 3, 4, 6, 8}, {2, 3});
  const std::vector<int64_t> shape = {3};
  const Handle bare(tr_gen_layer_norm(x.t, shape.data(), 1, nullptr, nullptr,
                                      /*eps=*/1e-5, /*cudnn_enable=*/true));
  EXPECT_EQ(shape_of(bare.t), (std::vector<int64_t>{2, 3}));
  const std::vector<float> b = data_of(bare.t);
  EXPECT_NEAR(b.at(0), -1.2247F, 1e-4F);
  EXPECT_NEAR(b.at(1), 0.0F, 1e-4F);
  EXPECT_NEAR(b.at(2), 1.2247F, 1e-4F);
  // affine: y = normalized * weight + bias, elementwise over the tail dim.
  const Handle weight = make({2.0F, 2.0F, 2.0F}, {3});
  const Handle bias = make({1.0F, 1.0F, 1.0F}, {3});
  const Handle affine(
      tr_gen_layer_norm(x.t, shape.data(), 1, weight.t, bias.t, 1e-5, true));
  const std::vector<float> a = data_of(affine.t);
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], 2.0F * b[i] + 1.0F, 1e-4F);
  }
  EXPECT_EQ(
      tr_gen_layer_norm(nullptr, shape.data(), 1, nullptr, nullptr, 1e-5, true),
      nullptr);
  expect_error_from("tr_gen_layer_norm");
}

// ---- masked_fill: bool-mask contract + null guard ----------------------

TEST(GeneratedTranche3, MaskedFillRequiresBoolMask) {
  const Handle x = make({10.0F, 20.0F, 30.0F, 40.0F}, {4});
  // Build the mask through the generated compare family — this is the
  // round trip the causal-mask idiom uses: comparisons yield genuine bool
  // handles (only the read path floatifies).
  const Handle picker = make({0.0F, 1.0F, 0.0F, 1.0F}, {4});
  const Handle mask(tr_gen_ne_scalar(picker.t, 0.0));
  const Handle filled(tr_gen_masked_fill_scalar(x.t, mask.t, /*value=*/-100.0));
  EXPECT_EQ(data_of(filled.t),
            (std::vector<float>{10.0F, -100.0F, 30.0F, -100.0F}));
  // A float tensor is not a valid mask: ATen demands bool (torch 2.x
  // dropped byte-mask tolerance), surfacing as NULL + error, not an abort.
  EXPECT_EQ(tr_gen_masked_fill_scalar(x.t, picker.t, -100.0), nullptr);
  expect_error_from("tr_gen_masked_fill_scalar");
  EXPECT_EQ(tr_gen_masked_fill_scalar(nullptr, mask.t, -100.0), nullptr);
  expect_error_from("tr_gen_masked_fill_scalar");
}

// ---- tril/triu: diagonal convention + null guards ----------------------

TEST(GeneratedTranche3, TrilTriuDiagonals) {
  const Handle m = make({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
  const Handle lower(tr_gen_tril(m.t, /*diagonal=*/0));
  EXPECT_EQ(data_of(lower.t), (std::vector<float>{1, 0, 0, 4, 5, 0, 7, 8, 9}));
  // diagonal=-1 excludes the main diagonal (strictly-below).
  const Handle strict(tr_gen_tril(m.t, -1));
  EXPECT_EQ(data_of(strict.t), (std::vector<float>{0, 0, 0, 4, 0, 0, 7, 8, 0}));
  const Handle upper(tr_gen_triu(m.t, 0));
  EXPECT_EQ(data_of(upper.t), (std::vector<float>{1, 2, 3, 0, 5, 6, 0, 0, 9}));
  // diagonal=+1 excludes the main diagonal (strictly-above).
  const Handle above(tr_gen_triu(m.t, 1));
  EXPECT_EQ(data_of(above.t), (std::vector<float>{0, 2, 3, 0, 0, 6, 0, 0, 0}));
  EXPECT_EQ(tr_gen_tril(nullptr, 0), nullptr);
  expect_error_from("tr_gen_tril");
  EXPECT_EQ(tr_gen_triu(nullptr, 0), nullptr);
  expect_error_from("tr_gen_triu");
}

}  // namespace
