#include <gtest/gtest.h>

#include <cmath>
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

std::vector<int64_t> indices_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data_i64(t, 0, nullptr, &numel), 2)
      << tr_last_error();
  std::vector<int64_t> out(numel);
  EXPECT_EQ(tr_tensor_copy_data_i64(t, numel, out.data(), &numel), 0)
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

void expect_near(const std::vector<float>& got, const std::vector<float>& want,
                 float tol) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
  }
}

void expect_error_from(const char* who) {
  const char* message = tr_last_error();
  ASSERT_NE(message, nullptr);
  EXPECT_NE(std::strstr(message, who), nullptr) << message;
}

TEST(GeneratedTranche7, TopkWritesValuesAndIndicesThroughItsOutPointers) {
  const Handle input = make({1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 6.0F}, {2, 3});
  tr_tensor* values = nullptr;
  tr_tensor* indices = nullptr;
  ASSERT_EQ(tr_gen_topk(input.t, 2, -1, true, true, &values, &indices), 0)
      << tr_last_error();
  const Handle owned_values(values);
  const Handle owned_indices(indices);
  EXPECT_EQ(shape_of(values), (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(data_of(values), (std::vector<float>{5.0F, 3.0F, 6.0F, 4.0F}));
  EXPECT_EQ(indices_of(indices), (std::vector<int64_t>{1, 2, 2, 0}));
}

TEST(GeneratedTranche7, SortWritesValuesAndIndicesThroughItsOutPointers) {
  const Handle input = make({3.0F, 1.0F, 2.0F}, {3});
  tr_tensor* values = nullptr;
  tr_tensor* indices = nullptr;
  ASSERT_EQ(tr_gen_sort(input.t, 0, true, &values, &indices), 0)
      << tr_last_error();
  const Handle owned_values(values);
  const Handle owned_indices(indices);
  EXPECT_EQ(data_of(values), (std::vector<float>{3.0F, 2.0F, 1.0F}));
  EXPECT_EQ(indices_of(indices), (std::vector<int64_t>{0, 2, 1}));
}

TEST(GeneratedTranche7, AFailedCallLeavesEveryOutPointerNull) {
  const Handle input = make({1.0F, 2.0F, 3.0F}, {3});
  tr_tensor* values = input.t;
  tr_tensor* indices = input.t;
  EXPECT_EQ(tr_gen_topk(input.t, 4, 0, true, true, &values, &indices), 1);
  expect_error_from("tr_gen_topk");
  EXPECT_EQ(values, nullptr);
  EXPECT_EQ(indices, nullptr);
}

TEST(GeneratedTranche7, ARefusedCallNullsEveryOutSlotItCanReach) {
  const Handle input = make({1.0F, 2.0F, 3.0F}, {3});
  tr_tensor* values = input.t;
  tr_tensor* indices = input.t;
  EXPECT_EQ(tr_gen_topk(nullptr, 1, 0, true, true, &values, &indices), 1);
  expect_error_from("tr_gen_topk");
  EXPECT_EQ(values, nullptr);
  EXPECT_EQ(indices, nullptr);
  indices = input.t;
  EXPECT_EQ(tr_gen_topk(input.t, 1, 0, true, true, nullptr, &indices), 1);
  expect_error_from("tr_gen_topk");
  EXPECT_EQ(indices, nullptr);
  values = input.t;
  EXPECT_EQ(tr_gen_sort(input.t, 0, false, &values, nullptr), 1);
  expect_error_from("tr_gen_sort");
  EXPECT_EQ(values, nullptr);
}

Handle full(const std::vector<int64_t>& dims, double value) {
  return Handle(tr_full(dims.data(), static_cast<int64_t>(dims.size()), value));
}

// Zero weights leave every gate at sigmoid(0) = 0.5 and the candidate at
// tanh(0) = 0, so the recurrences reduce to closed forms of the state.
struct ZeroWeights {
  Handle w_ih, w_hh, b_ih, b_hh;
  ZeroWeights(int64_t gates, int64_t input, int64_t hidden)
      : w_ih(full({gates * hidden, input}, 0.0)),
        w_hh(full({gates * hidden, hidden}, 0.0)),
        b_ih(full({gates * hidden}, 0.0)),
        b_hh(full({gates * hidden}, 0.0)) {}
  std::vector<const tr_tensor*> list() const {
    return {w_ih.t, w_hh.t, b_ih.t, b_hh.t};
  }
};

TEST(GeneratedTranche7, ArgsortIsTheIndicesHalfOfSort) {
  const Handle input = make({3.0F, 1.0F, 2.0F}, {3});
  const Handle indices(tr_gen_argsort(input.t, 0, true));
  EXPECT_EQ(indices_of(indices.t), (std::vector<int64_t>{0, 2, 1}));
  EXPECT_EQ(tr_gen_argsort(nullptr, 0, true), nullptr);
  expect_error_from("tr_gen_argsort");
}

TEST(GeneratedTranche7, MultinomialDrawsFromTheGeneratorItIsHanded) {
  const Handle weights = make({0.0F, 1.0F, 0.0F, 0.5F, 0.0F, 0.5F}, {2, 3});
  tr_generator* first = tr_generator_new(7);
  tr_generator* second = tr_generator_new(7);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const Handle a(tr_gen_multinomial(weights.t, 8, true, first));
  const Handle b(tr_gen_multinomial(weights.t, 8, true, second));
  tr_generator_free(first);
  tr_generator_free(second);
  EXPECT_EQ(shape_of(a.t), (std::vector<int64_t>{2, 8}));
  const std::vector<int64_t> draws = indices_of(a.t);
  EXPECT_EQ(draws, indices_of(b.t));
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(draws[i], 1) << "row 0, draw " << i;
    EXPECT_NE(draws[8 + i], 1) << "row 1, draw " << i;
  }
  const Handle global(tr_gen_multinomial(weights.t, 1, false, nullptr));
  EXPECT_EQ(shape_of(global.t), (std::vector<int64_t>{2, 1}));
  EXPECT_EQ(tr_gen_multinomial(nullptr, 1, false, nullptr), nullptr);
  expect_error_from("tr_gen_multinomial");
}

TEST(GeneratedTranche7, LstmWithZeroWeightsHalvesTheCellEachStep) {
  const Handle input = full({2, 1, 3}, 1.0);
  const Handle h0 = full({1, 1, 4}, 0.0);
  const Handle c0 = full({1, 1, 4}, 1.0);
  const ZeroWeights weights(4, 3, 4);
  const std::vector<const tr_tensor*> state = {h0.t, c0.t};
  const std::vector<const tr_tensor*> params = weights.list();
  tr_tensor* output = nullptr;
  tr_tensor* h_n = nullptr;
  tr_tensor* c_n = nullptr;
  ASSERT_EQ(tr_gen_lstm_input(input.t, state.data(), 2, params.data(), 4, true,
                              1, 0.0, false, false, false, &output, &h_n, &c_n),
            0)
      << tr_last_error();
  const Handle owned_output(output);
  const Handle owned_h(h_n);
  const Handle owned_c(c_n);
  EXPECT_EQ(shape_of(output), (std::vector<int64_t>{2, 1, 4}));
  EXPECT_EQ(shape_of(h_n), (std::vector<int64_t>{1, 1, 4}));
  expect_near(data_of(c_n), std::vector<float>(4, 0.25F), 1e-6F);
  expect_near(data_of(h_n), std::vector<float>(4, 0.5F * std::tanh(0.25F)),
              1e-6F);
  const std::vector<float> steps = data_of(output);
  EXPECT_NEAR(steps[0], 0.5F * std::tanh(0.5F), 1e-6F);
  EXPECT_NEAR(steps[4], 0.5F * std::tanh(0.25F), 1e-6F);
}

TEST(GeneratedTranche7, GruWithZeroWeightsHalvesTheStateEachStep) {
  const Handle input = full({3, 1, 2}, 1.0);
  const Handle h0 = full({1, 1, 2}, 1.0);
  const ZeroWeights weights(3, 2, 2);
  const std::vector<const tr_tensor*> params = weights.list();
  tr_tensor* output = nullptr;
  tr_tensor* h_n = nullptr;
  ASSERT_EQ(tr_gen_gru_input(input.t, h0.t, params.data(), 4, true, 1, 0.0,
                             false, false, false, &output, &h_n),
            0)
      << tr_last_error();
  const Handle owned_output(output);
  const Handle owned_h(h_n);
  expect_near(data_of(output), {0.5F, 0.5F, 0.25F, 0.25F, 0.125F, 0.125F},
              1e-6F);
  expect_near(data_of(h_n), {0.125F, 0.125F}, 1e-6F);
}

TEST(GeneratedTranche7, RecurrencesRefuseNullListsAndNullListElements) {
  const Handle input = full({1, 1, 2}, 1.0);
  const Handle h0 = full({1, 1, 2}, 0.0);
  const ZeroWeights weights(3, 2, 2);
  std::vector<const tr_tensor*> params = weights.list();
  tr_tensor* output = nullptr;
  tr_tensor* h_n = nullptr;
  EXPECT_EQ(tr_gen_gru_input(input.t, h0.t, nullptr, 4, true, 1, 0.0, false,
                             false, false, &output, &h_n),
            1);
  expect_error_from("tr_gen_gru_input");
  params[2] = nullptr;
  EXPECT_EQ(tr_gen_gru_input(input.t, h0.t, params.data(), 4, true, 1, 0.0,
                             false, false, false, &output, &h_n),
            1);
  expect_error_from("tr_gen_gru_input");
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(h_n, nullptr);
  tr_tensor* c_n = nullptr;
  EXPECT_EQ(tr_gen_lstm_input(input.t, nullptr, 2, params.data(), 4, true, 1,
                              0.0, false, false, false, &output, &h_n, &c_n),
            1);
  expect_error_from("tr_gen_lstm_input");
}

TEST(GeneratedTranche7, FlattenWeightGuardsItsListAndHasNoCpuKernel) {
  EXPECT_EQ(tr_gen__cudnn_rnn_flatten_weight(nullptr, 4, 4, 2, 3, 2, 0, 1,
                                             false, false),
            nullptr);
  expect_error_from("tr_gen__cudnn_rnn_flatten_weight");
  if (tr_cuda_is_available()) {
    GTEST_SKIP() << "the CUDA path is driven from the Racket layer tests";
  }
  const ZeroWeights weights(3, 2, 2);
  const std::vector<const tr_tensor*> params = weights.list();
  EXPECT_EQ(tr_gen__cudnn_rnn_flatten_weight(params.data(), 4, 4, 2, 3, 2, 0, 1,
                                             false, false),
            nullptr);
  expect_error_from("tr_gen__cudnn_rnn_flatten_weight");
}

}  // namespace
