#include "torchrkt/c_api/audio.h"

#include <sndfile.h>
#include <torch/torch.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "torchrkt/detail/op_call.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

namespace {

struct SndClose {
  void operator()(SNDFILE* f) const noexcept {
    if (f != nullptr) {
      sf_close(f);
    }
  }
};

using SndPtr = std::unique_ptr<SNDFILE, SndClose>;

SndPtr open_for_read(const char* path, SF_INFO* info) {
  std::memset(info, 0, sizeof(*info));
  SndPtr f(sf_open(path, SFM_READ, info));
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path + ": " +
                             sf_strerror(nullptr));
  }
  return f;
}

int format_for(const std::string& path) {
  const auto dot = path.rfind('.');
  const std::string ext = dot == std::string::npos ? "" : path.substr(dot + 1);
  if (ext == "wav") {
    return SF_FORMAT_WAV | SF_FORMAT_PCM_16;
  }
  if (ext == "flac") {
    return SF_FORMAT_FLAC | SF_FORMAT_PCM_16;
  }
  throw std::invalid_argument("unsupported audio extension ." + ext +
                              " (wav and flac)");
}

}  // namespace

extern "C" {

int tr_audio_info(const char* path, int64_t* frames, int32_t* rate,
                  int32_t* channels) {
  if (path == nullptr || frames == nullptr || rate == nullptr ||
      channels == nullptr) {
    return torchrkt::null_arg_status("tr_audio_info");
  }
  return torchrkt::status_call("tr_audio_info", [&] {
    SF_INFO info;
    const SndPtr f = open_for_read(path, &info);
    *frames = info.frames;
    *rate = info.samplerate;
    *channels = info.channels;
  });
}

tr_tensor* tr_audio_load(const char* path, int64_t frame_offset,
                         int64_t num_frames, int32_t* rate) {
  if (path == nullptr || rate == nullptr) {
    return torchrkt::null_arg("tr_audio_load");
  }
  return torchrkt::alloc_result("tr_audio_load", [&] {
    SF_INFO info;
    const SndPtr f = open_for_read(path, &info);
    *rate = info.samplerate;
    if (num_frames < -1) {
      throw std::invalid_argument(
          "num_frames must be -1 (to end) or "
          "nonnegative, got " +
          std::to_string(num_frames));
    }
    if (frame_offset < 0 || frame_offset > info.frames) {
      throw std::out_of_range("frame offset " + std::to_string(frame_offset) +
                              " outside " + std::to_string(info.frames) +
                              " frames");
    }
    const int64_t remaining = info.frames - frame_offset;
    const int64_t wanted =
        num_frames < 0 ? remaining : std::min(num_frames, remaining);
    if (frame_offset > 0 &&
        sf_seek(f.get(), frame_offset, SEEK_SET) != frame_offset) {
      throw std::runtime_error(std::string("seek failed: ") +
                               sf_strerror(f.get()));
    }
    torch::Tensor interleaved =
        torch::empty({wanted, info.channels}, torch::kFloat32);
    const sf_count_t got =
        wanted == 0
            ? 0
            : sf_readf_float(f.get(), interleaved.data_ptr<float>(), wanted);
    if (got != wanted) {
      throw std::runtime_error("short read: wanted " + std::to_string(wanted) +
                               " frames, got " + std::to_string(got));
    }
    return interleaved.t().contiguous();
  });
}

int tr_audio_save(const char* path, const tr_tensor* samples, int32_t rate) {
  if (path == nullptr || samples == nullptr) {
    return torchrkt::null_arg_status("tr_audio_save");
  }
  return torchrkt::status_call("tr_audio_save", [&] {
    torch::Tensor s = samples->value;
    if (s.dim() == 1) {
      s = s.unsqueeze(0);
    }
    if (s.dim() != 2 || s.size(0) < 1) {
      throw std::invalid_argument("samples must be rank 1 or (channels n)");
    }
    if (rate < 1) {
      throw std::invalid_argument("sample rate must be positive");
    }
    const torch::Tensor interleaved =
        s.to(torch::kCPU).to(torch::kFloat32).t().contiguous();
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    info.samplerate = rate;
    info.channels = static_cast<int>(s.size(0));
    info.format = format_for(path);
    if (sf_format_check(&info) == 0) {
      throw std::invalid_argument(
          "libsndfile rejects this format/channel "
          "combination");
    }
    SndPtr f(sf_open(path, SFM_WRITE, &info));
    if (!f) {
      throw std::runtime_error(std::string("cannot open ") + path + ": " +
                               sf_strerror(nullptr));
    }
    // float->PCM16 wraps on overflow by default; clamp like write-wav
    sf_command(f.get(), SFC_SET_CLIPPING, nullptr, SF_TRUE);
    const sf_count_t frames = s.size(1);
    const sf_count_t wrote =
        frames == 0
            ? 0
            : sf_writef_float(f.get(), interleaved.data_ptr<float>(), frames);
    if (wrote != frames) {
      throw std::runtime_error("short write: wanted " + std::to_string(frames) +
                               " frames, wrote " + std::to_string(wrote));
    }
    // the header is finalized at close; a flush failure must not report
    // success, so close explicitly and surface the result
    SNDFILE* raw = f.release();
    if (const int rc = sf_close(raw); rc != 0) {
      throw std::runtime_error(std::string("close failed: ") +
                               sf_error_number(rc));
    }
  });
}

}  // extern "C"
