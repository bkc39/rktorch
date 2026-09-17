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

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

tr_dtype dtype_of(const tr_tensor* t) {
  tr_dtype dt = TR_DTYPE_KEEP;
  EXPECT_EQ(tr_tensor_dtype(t, &dt), 0) << tr_last_error();
  return dt;
}

std::vector<float> data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, 0, nullptr, &numel), 2) << tr_last_error();
  std::vector<float> out(numel);
  EXPECT_EQ(tr_tensor_copy_data(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

std::vector<uint8_t> bytes_of(const tr_tensor* t) {
  std::uint64_t nbytes = 0;
  EXPECT_EQ(tr_tensor_copy_bytes(t, 0, nullptr, &nbytes), 2) << tr_last_error();
  std::vector<uint8_t> out(nbytes);
  EXPECT_EQ(tr_tensor_copy_bytes(t, nbytes, out.data(), &nbytes), 0)
      << tr_last_error();
  return out;
}

TEST(TorchrktDtype, HalfPairRoundTripsThroughFloat32Readback) {
  const Handle x = make({1.0F, 0.1F, -2.5F, 65504.0F}, {4});
  const Handle half(tr_tensor_to_dtype(x.t, TR_DTYPE_FLOAT16));
  EXPECT_EQ(dtype_of(half.t), TR_DTYPE_FLOAT16);
  const std::vector<float> h = data_of(half.t);
  EXPECT_FLOAT_EQ(h[0], 1.0F);
  EXPECT_NEAR(h[1], 0.1F, 1e-4F);
  EXPECT_NE(h[1], 0.1F) << "float16 has ten mantissa bits";
  EXPECT_FLOAT_EQ(h[2], -2.5F);
  EXPECT_FLOAT_EQ(h[3], 65504.0F);
  const Handle brain(tr_tensor_to_dtype(x.t, TR_DTYPE_BFLOAT16));
  EXPECT_EQ(dtype_of(brain.t), TR_DTYPE_BFLOAT16);
  const std::vector<float> b = data_of(brain.t);
  EXPECT_FLOAT_EQ(b[0], 1.0F);
  EXPECT_NEAR(b[1], 0.1F, 1e-3F);
  EXPECT_FLOAT_EQ(b[2], -2.5F);
  EXPECT_NEAR(b[3], 65504.0F, 256.0F) << "bfloat16 keeps float32's range";
  const Handle back(tr_tensor_to_dtype(half.t, TR_DTYPE_FLOAT32));
  EXPECT_EQ(dtype_of(back.t), TR_DTYPE_FLOAT32);
  const Handle three = make({3.0F}, {1});
  const Handle scalar(tr_tensor_to_dtype(three.t, TR_DTYPE_BFLOAT16));
  double item = 0.0;
  ASSERT_EQ(tr_tensor_item(scalar.t, &item), 0) << tr_last_error();
  EXPECT_DOUBLE_EQ(item, 3.0);
}

TEST(TorchrktDtype, CopyBytesIsTheElementBytesInTheOwnDtype) {
  const Handle x = make({1.0F, -2.0F}, {2});
  const Handle half(tr_tensor_to_dtype(x.t, TR_DTYPE_FLOAT16));
  // IEEE half: 1.0 is 0x3C00, -2.0 is 0xC000, little-endian
  EXPECT_EQ(bytes_of(half.t), (std::vector<uint8_t>{0x00, 0x3C, 0x00, 0xC0}));
  const Handle brain(tr_tensor_to_dtype(x.t, TR_DTYPE_BFLOAT16));
  EXPECT_EQ(bytes_of(brain.t), (std::vector<uint8_t>{0x80, 0x3F, 0x00, 0xC0}));
  const std::vector<uint8_t> f32 = bytes_of(x.t);
  ASSERT_EQ(f32.size(), 8U);
  float first = 0.0F;
  std::memcpy(&first, f32.data(), sizeof first);
  EXPECT_EQ(first, 1.0F);
  const Handle mask(tr_gen_gt_scalar(x.t, 0.0));
  EXPECT_EQ(bytes_of(mask.t), (std::vector<uint8_t>{1, 0}));
  const Handle i64(tr_tensor_to_dtype(x.t, TR_DTYPE_INT64));
  EXPECT_EQ(bytes_of(i64.t).size(), 16U);
  std::uint64_t nbytes = 7;
  EXPECT_EQ(tr_tensor_copy_bytes(nullptr, 0, nullptr, &nbytes), 1);
  EXPECT_EQ(tr_tensor_copy_bytes(x.t, 0, nullptr, nullptr), 1);
}

TEST(TorchrktDtype, FromBytesInvertsCopyBytesForEveryDtype) {
  const std::vector<int64_t> dims = {2};
  const std::vector<uint8_t> half = {0x00, 0x3C, 0x00, 0xC0};
  const Handle h(tr_from_bytes(half.data(), half.size(), dims.data(), 1,
                               TR_DTYPE_FLOAT16));
  EXPECT_EQ(dtype_of(h.t), TR_DTYPE_FLOAT16);
  EXPECT_EQ(data_of(h.t), (std::vector<float>{1.0F, -2.0F}));
  const std::vector<uint8_t> brain = {0x80, 0x3F, 0x00, 0xC0};
  const Handle b(tr_from_bytes(brain.data(), brain.size(), dims.data(), 1,
                               TR_DTYPE_BFLOAT16));
  EXPECT_EQ(dtype_of(b.t), TR_DTYPE_BFLOAT16);
  EXPECT_EQ(data_of(b.t), (std::vector<float>{1.0F, -2.0F}));
  const Handle x = make({1.5F, -0.25F}, {2});
  for (const tr_dtype dt :
       {TR_DTYPE_FLOAT32, TR_DTYPE_FLOAT64, TR_DTYPE_INT64, TR_DTYPE_UINT8,
        TR_DTYPE_BOOL, TR_DTYPE_FLOAT16, TR_DTYPE_BFLOAT16}) {
    const Handle cast(tr_tensor_to_dtype(x.t, dt));
    const std::vector<uint8_t> raw = bytes_of(cast.t);
    const Handle again(
        tr_from_bytes(raw.data(), raw.size(), dims.data(), 1, dt));
    EXPECT_EQ(dtype_of(again.t), dt);
    EXPECT_EQ(data_of(again.t), data_of(cast.t)) << "dtype " << dt;
  }
  const std::vector<int64_t> none = {0};
  const Handle empty(
      tr_from_bytes(nullptr, 0, none.data(), 1, TR_DTYPE_FLOAT16));
  EXPECT_EQ(dtype_of(empty.t), TR_DTYPE_FLOAT16);
  // three bytes are not two halves
  EXPECT_EQ(tr_from_bytes(half.data(), 3, dims.data(), 1, TR_DTYPE_FLOAT16),
            nullptr);
  EXPECT_NE(std::strstr(tr_last_error(), "element size"), nullptr)
      << tr_last_error();
  EXPECT_EQ(tr_from_bytes(half.data(), 4, dims.data(), 1, TR_DTYPE_KEEP),
            nullptr);
  EXPECT_EQ(tr_from_bytes(nullptr, 4, dims.data(), 1, TR_DTYPE_FLOAT16),
            nullptr);
}

}  // namespace
