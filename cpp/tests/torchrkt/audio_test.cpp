#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
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

Handle make(const std::vector<float>& values,
            const std::vector<int64_t>& dims) {
  return Handle(tr_from_data(values.data(), values.size(), dims.data(),
                             static_cast<int64_t>(dims.size())));
}

std::string temp_audio_path(const char* name) {
  return testing::TempDir() + name;
}

class AudioRoundTrip : public testing::TestWithParam<const char*> {};

TEST_P(AudioRoundTrip, SaveInfoLoad) {
  const std::string path = temp_audio_path(GetParam());
  const Handle stereo =
      make({0.5F, -0.25F, 0.125F, -1.0F, 0.0F, 0.75F}, {2, 3});
  ASSERT_EQ(tr_audio_save(path.c_str(), stereo.t, 22050), 0) << tr_last_error();

  int64_t frames = 0;
  int32_t rate = 0;
  int32_t channels = 0;
  ASSERT_EQ(tr_audio_info(path.c_str(), &frames, &rate, &channels), 0)
      << tr_last_error();
  EXPECT_EQ(frames, 3);
  EXPECT_EQ(rate, 22050);
  EXPECT_EQ(channels, 2);

  int32_t load_rate = 0;
  const Handle back(tr_audio_load(path.c_str(), 0, -1, &load_rate));
  EXPECT_EQ(load_rate, 22050);
  EXPECT_EQ(data_of(back.t), data_of(stereo.t));
  std::remove(path.c_str());
}

INSTANTIATE_TEST_SUITE_P(WavAndFlac, AudioRoundTrip,
                         testing::Values("rt.wav", "rt.flac"));

TEST(Audio, WindowedLoadMatchesFullSlice) {
  const std::string path = temp_audio_path("window.wav");
  const Handle mono = make({0.125F, 0.25F, 0.375F, 0.5F, 0.625F}, {1, 5});
  ASSERT_EQ(tr_audio_save(path.c_str(), mono.t, 8000), 0) << tr_last_error();
  int32_t wr = 0;
  const Handle window(tr_audio_load(path.c_str(), 1, 3, &wr));
  EXPECT_EQ(data_of(window.t), (std::vector<float>{0.25F, 0.375F, 0.5F}));
  const Handle tail(tr_audio_load(path.c_str(), 3, -1, &wr));
  EXPECT_EQ(data_of(tail.t), (std::vector<float>{0.5F, 0.625F}));
  std::remove(path.c_str());
}

TEST(Audio, RejectionsAndGuards) {
  const std::string missing = temp_audio_path("missing.wav");
  int64_t frames = 0;
  int32_t rate = 0;
  int32_t channels = 0;
  EXPECT_EQ(tr_audio_info(missing.c_str(), &frames, &rate, &channels), 1);
  EXPECT_NE(tr_last_error(), nullptr);
  int32_t gr = 0;
  EXPECT_EQ(tr_audio_load(missing.c_str(), 0, -1, &gr), nullptr);

  const std::string bad_ext = temp_audio_path("clip.mp3");
  const Handle mono = make({0.5F}, {1, 1});
  EXPECT_EQ(tr_audio_save(bad_ext.c_str(), mono.t, 8000), 1);

  const std::string path = temp_audio_path("guards.wav");
  ASSERT_EQ(tr_audio_save(path.c_str(), mono.t, 8000), 0) << tr_last_error();
  EXPECT_EQ(tr_audio_load(path.c_str(), 5, -1, &gr), nullptr);
  EXPECT_EQ(tr_audio_load(path.c_str(), 0, -2, &gr), nullptr);
  EXPECT_EQ(tr_audio_save(path.c_str(), mono.t, 0), 1);

  const Handle scalar = make({0.5F}, {});
  EXPECT_EQ(tr_audio_save(path.c_str(), scalar.t, 8000), 1);
  const Handle cube = make({0.5F, 0.5F}, {1, 2, 1});
  EXPECT_EQ(tr_audio_save(path.c_str(), cube.t, 8000), 1);
  EXPECT_EQ(tr_audio_info(nullptr, &frames, &rate, &channels), 1);
  EXPECT_EQ(tr_audio_load(nullptr, 0, -1, &gr), nullptr);
  EXPECT_EQ(tr_audio_load(path.c_str(), 0, -1, nullptr), nullptr);
  EXPECT_EQ(tr_audio_save(nullptr, mono.t, 8000), 1);
  EXPECT_EQ(tr_audio_save(path.c_str(), nullptr, 8000), 1);
  std::remove(path.c_str());
}

}  // namespace
