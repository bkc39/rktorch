#include <gtest/gtest.h>

#include <cstdint>
#include <string>
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
  int64_t numel = 0;
  EXPECT_EQ(tr_tensor_numel(t, &numel), 0) << tr_last_error();
  std::vector<float> out(static_cast<size_t>(numel));
  uint64_t got = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, out.size(), out.data(), &got), 0)
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

TEST(Spectral, HannWindowMatchesClosedForm) {
  const Handle w(tr_hann_window(4, true));
  EXPECT_EQ(data_of(w.t), (std::vector<float>{0.0F, 0.5F, 1.0F, 0.5F}));
  const Handle symmetric(tr_hann_window(4, false));
  const std::vector<float> s = data_of(symmetric.t);
  EXPECT_FLOAT_EQ(s[0], 0.0F);
  EXPECT_FLOAT_EQ(s[3], 0.0F);
  const Handle empty(tr_hann_window(0, true));
  EXPECT_EQ(data_of(empty.t), std::vector<float>{});
  EXPECT_EQ(tr_hann_window(-1, true), nullptr);
  EXPECT_NE(tr_last_error(), nullptr);
}

TEST(Spectral, StftConstantSignalConcentratesAtDc) {
  const Handle ones =
      make({1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F}, {8});
  const Handle out(tr_stft(ones.t, 4, 4, -1, nullptr, /*center=*/false,
                           /*normalized=*/false));
  EXPECT_EQ(shape_of(out.t), (std::vector<int64_t>{3, 2, 2}));
  const std::vector<float> v = data_of(out.t);
  EXPECT_FLOAT_EQ(v[0], 4.0F);
  EXPECT_FLOAT_EQ(v[1], 0.0F);
  for (size_t i = 4; i < v.size(); ++i) {
    EXPECT_FLOAT_EQ(v[i], 0.0F) << "bin element " << i;
  }
}

TEST(Spectral, StftHonorsWindowAndGuards) {
  const Handle signal = make({1.0F, 1.0F, 1.0F, 1.0F}, {4});
  const Handle window(tr_hann_window(4, true));
  const Handle out(tr_stft(signal.t, 4, 4, 4, window.t, false, false));
  const std::vector<float> v = data_of(out.t);
  EXPECT_FLOAT_EQ(v[0], 2.0F);
  EXPECT_EQ(tr_stft(nullptr, 4, -1, -1, nullptr, true, false), nullptr);
  EXPECT_NE(tr_last_error(), nullptr);
  EXPECT_EQ(tr_stft(signal.t, 0, -1, -1, nullptr, true, false), nullptr);
  EXPECT_NE(std::string(tr_last_error()).find("n_fft"), std::string::npos);
  EXPECT_EQ(tr_stft(signal.t, 4, -2, -1, nullptr, false, false), nullptr);
  EXPECT_EQ(tr_stft(signal.t, 4, 0, -1, nullptr, false, false), nullptr);
  EXPECT_EQ(tr_stft(signal.t, 4, -1, -2, nullptr, false, false), nullptr);
  EXPECT_EQ(tr_stft(signal.t, 4, -1, 0, nullptr, false, false), nullptr);
}

TEST(Spectral, StftDefaultWindowMatchesInputDtype) {
  const Handle ones =
      make({1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F}, {8});
  const Handle wide(tr_tensor_to_dtype(ones.t, TR_DTYPE_FLOAT64));
  const Handle out(tr_stft(wide.t, 4, 4, -1, nullptr, /*center=*/false,
                           /*normalized=*/false));
  tr_dtype dtype = TR_DTYPE_FLOAT32;
  EXPECT_EQ(tr_tensor_dtype(out.t, &dtype), 0) << tr_last_error();
  EXPECT_EQ(dtype, TR_DTYPE_FLOAT64);
  EXPECT_EQ(shape_of(out.t), (std::vector<int64_t>{3, 2, 2}));
}

}  // namespace
