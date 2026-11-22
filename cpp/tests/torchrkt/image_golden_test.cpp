#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "test_data.hpp"
#include "torchrkt/c_api.h"

namespace {

constexpr std::array<uint32_t, 64> kRound = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

uint32_t rotr(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

void compress(std::array<uint32_t, 8>& h, const uint8_t* block) {
  std::array<uint32_t, 64> w{};
  for (size_t i = 0; i < 16; ++i) {
    w[i] = (uint32_t{block[4 * i]} << 24) |
           (uint32_t{block[(4 * i) + 1]} << 16) |
           (uint32_t{block[(4 * i) + 2]} << 8) | block[(4 * i) + 3];
  }
  for (size_t i = 16; i < 64; ++i) {
    const uint32_t s0 =
        rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 =
        rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::array<uint32_t, 8> v = h;
  for (size_t i = 0; i < 64; ++i) {
    const uint32_t t1 = v[7] +
                        (rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25)) +
                        ((v[4] & v[5]) ^ (~v[4] & v[6])) + kRound[i] + w[i];
    const uint32_t t2 = (rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22)) +
                        ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
    v = {t1 + t2, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
  }
  for (size_t i = 0; i < 8; ++i) {
    h[i] += v[i];
  }
}

std::string sha256(std::vector<uint8_t> bytes) {
  const uint64_t bits = uint64_t{bytes.size()} * 8;
  bytes.push_back(0x80);
  while (bytes.size() % 64 != 56) {
    bytes.push_back(0);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<uint8_t>(bits >> shift));
  }
  std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  for (size_t at = 0; at < bytes.size(); at += 64) {
    compress(h, bytes.data() + at);
  }
  std::string hex;
  for (const uint32_t word : h) {
    std::array<char, 9> digits{};
    std::snprintf(digits.data(), digits.size(), "%08x", word);
    hex += digits.data();
  }
  return hex;
}

struct Handle {
  tr_tensor* t;
  explicit Handle(tr_tensor* p) : t(p) {}
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() {
    tr_tensor_free(t);
  }
};

tr_tensor* decode(const std::string& name, int32_t channels) {
  const std::vector<uint8_t>& bytes = embedded_files().at(name);
  return tr_image_decode(bytes.data(), static_cast<int64_t>(bytes.size()),
                         channels);
}

std::vector<int64_t> shape_of(const tr_tensor* t) {
  std::vector<int64_t> dims(3);
  int64_t got = 0;
  EXPECT_EQ(tr_tensor_shape(t, 3, dims.data(), &got), 0) << tr_last_error();
  return dims;
}

std::vector<uint8_t> bytes_of(const tr_tensor* t) {
  int64_t numel = 0;
  EXPECT_EQ(tr_tensor_numel(t, &numel), 0) << tr_last_error();
  std::vector<uint8_t> out(static_cast<size_t>(numel));
  uint64_t got = 0;
  EXPECT_EQ(tr_tensor_copy_data_u8(t, out.size(), out.data(), &got), 0)
      << tr_last_error();
  return out;
}

TEST(ImageDecodeGolden, Sha256MatchesKnownDigests) {
  EXPECT_EQ(sha256({}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256(std::vector<uint8_t>(64, 'a')),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
}

TEST(ImageDecodeGolden, JpegsDecodeToTorchvisionsPixels) {
  const std::map<std::string, int32_t> channels = {
      {"unchanged", 0}, {"gray", 1}, {"rgb", 3}};
  const std::vector<uint8_t>& text = embedded_files().at("image-goldens.txt");
  std::istringstream lines(std::string(text.begin(), text.end()));
  std::string name;
  std::string mode;
  std::vector<int64_t> shape(3);
  std::string digest;
  int cases = 0;
  while (lines >> name >> mode >> shape[0] >> shape[1] >> shape[2] >> digest) {
    const Handle img(decode(name, channels.at(mode)));
    ASSERT_NE(img.t, nullptr) << name << " " << mode << ": " << tr_last_error();
    EXPECT_EQ(shape_of(img.t), shape) << name << " " << mode;
    EXPECT_EQ(sha256(bytes_of(img.t)), digest) << name << " " << mode;
    ++cases;
  }
  EXPECT_EQ(cases, 6);
}

TEST(ImageDecodeGolden, MalformedJpegsAreErrors) {
  EXPECT_EQ(decode("bad-precision.jpg", 0), nullptr);
  EXPECT_NE(std::string(tr_last_error()).find("cannot decode image"),
            std::string::npos)
      << tr_last_error();
  EXPECT_EQ(decode("huge.jpg", 0), nullptr);
  EXPECT_NE(std::string(tr_last_error())
                .find("16000 x 16000 pixels is over the 178956970-pixel limit"),
            std::string::npos)
      << tr_last_error();
}

}  // namespace
