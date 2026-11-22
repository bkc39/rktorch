#include "torchrkt/detail/jpeg.hpp"

// jpeglib.h uses FILE and size_t without including their headers
// clang-format off
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>
// clang-format on

#include <algorithm>
#include <array>
#include <csetjmp>
#include <cstring>
#include <vector>

namespace torchrkt::detail {

namespace {

struct ErrorManager {
  jpeg_error_mgr pub;
  std::jmp_buf jump;
  std::array<char, JMSG_LENGTH_MAX> message;
};

ErrorManager& manager_of(j_common_ptr cinfo) {
  return *reinterpret_cast<ErrorManager*>(cinfo->err);
}

[[noreturn]] void error_exit(j_common_ptr cinfo) {
  ErrorManager& err = manager_of(cinfo);
  (*cinfo->err->format_message)(cinfo, err.message.data());
  std::longjmp(err.jump, 1);
}

void silent(j_common_ptr /*cinfo*/) {}

// torchvision's source manager: running out of bytes is an error, where
// libjpeg's own pads the image with a fake EOI and a warning
[[noreturn]] boolean truncated(j_decompress_ptr cinfo) {
  ErrorManager& err = manager_of(reinterpret_cast<j_common_ptr>(cinfo));
  constexpr const char* why = "Image is incomplete or truncated";
  std::strncpy(err.message.data(), why, err.message.size() - 1);
  std::longjmp(err.jump, 1);
}

class Decompressor {
 public:
  Decompressor() {
    cinfo_.err = jpeg_std_error(&err_.pub);
    err_.pub.error_exit = error_exit;
    err_.pub.output_message = silent;
  }
  Decompressor(const Decompressor&) = delete;
  Decompressor& operator=(const Decompressor&) = delete;
  Decompressor(Decompressor&&) = delete;
  Decompressor& operator=(Decompressor&&) = delete;
  ~Decompressor() {
    jpeg_destroy_decompress(&cinfo_);
  }

  jpeg_decompress_struct& info() {
    return cinfo_;
  }

  // A libjpeg error longjmps back here, over f's frame and libjpeg's: f
  // may hold nothing with a destructor
  template <typename F>
  void run(F&& f) {
    if (setjmp(err_.jump) != 0) {
      throw std::runtime_error(std::string("cannot decode image: ") +
                               err_.message.data());
    }
    f(cinfo_);
  }

 private:
  jpeg_decompress_struct cinfo_{};
  ErrorManager err_{};
};

// torchvision's CMYK to RGB, after Pillow's: k - k * (255 - c) / 255
uint8_t from_cmyk(int k, int cmy) {
  int v = (k * (255 - cmy)) + 128;
  v = ((v >> 8) + v) >> 8;
  return static_cast<uint8_t>(std::clamp(k - v, 0, 255));
}

// torchvision's RGB to gray for CMYK, after Pillow's ITU-R 601-2 luma
uint8_t luma(int r, int g, int b) {
  return static_cast<uint8_t>(
      ((r * 19595) + (g * 38470) + (b * 7471) + 0x8000) >> 16);
}

// A row libjpeg cannot write directly: CMYK to gray or RGB, or gray with an
// opaque alpha, the alpha 255 in every case
void convert_row(const uint8_t* in, int in_channels, uint8_t* out,
                 int out_channels, size_t width) {
  for (size_t x = 0; x < width; ++x) {
    std::array<uint8_t, 3> rgb{in[0], in[0], in[0]};
    if (in_channels == 4) {
      for (size_t c = 0; c < 3; ++c) {
        rgb.at(c) = from_cmyk(in[3], in[c]);
      }
    }
    if (out_channels <= 2) {
      out[0] = in_channels == 4 ? luma(rgb[0], rgb[1], rgb[2]) : in[0];
    } else {
      std::copy(rgb.begin(), rgb.end(), out);
    }
    if (out_channels % 2 == 0) {
      out[out_channels - 1] = 255;
    }
    in += in_channels;
    out += out_channels;
  }
}

}  // namespace

torch::Tensor decode_jpeg(const uint8_t* data, size_t len, int32_t channels) {
  Decompressor d;
  d.run([&](jpeg_decompress_struct& cinfo) {
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(len));
    cinfo.src->fill_input_buffer = truncated;
    jpeg_read_header(&cinfo, TRUE);
  });
  jpeg_decompress_struct& info = d.info();
  check_pixels(info.image_width, info.image_height);
  const bool cmyk =
      info.jpeg_color_space == JCS_CMYK || info.jpeg_color_space == JCS_YCCK;
  const int stored = info.num_components == 1 ? 1 : 3;
  const int kept = channels != 0 ? channels : stored;
  const bool gray = kept <= 2;
  const bool direct = !cmyk && kept != 2;
  if (cmyk) {
    info.out_color_space = JCS_CMYK;
  } else if (gray) {
    info.out_color_space = JCS_GRAYSCALE;
  } else {
    info.out_color_space = kept == 4 ? JCS_EXT_RGBA : JCS_RGB;
  }
  d.run([](jpeg_decompress_struct& cinfo) { jpeg_start_decompress(&cinfo); });
  const auto height = static_cast<size_t>(info.output_height);
  const auto width = static_cast<size_t>(info.output_width);
  const size_t stride = width * static_cast<size_t>(kept);
  torch::Tensor pixels = torch::empty(
      {static_cast<int64_t>(height), static_cast<int64_t>(width), kept},
      torch::kUInt8);
  uint8_t* base = pixels.data_ptr<uint8_t>();
  if (direct) {
    std::vector<JSAMPROW> rows(height);
    for (size_t y = 0; y < height; ++y) {
      rows[y] = base + (y * stride);
    }
    d.run([&](jpeg_decompress_struct& cinfo) {
      while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, rows.data() + cinfo.output_scanline,
                            cinfo.output_height - cinfo.output_scanline);
      }
      jpeg_finish_decompress(&cinfo);
    });
    return pixels;
  }
  const int components = info.output_components;
  std::vector<uint8_t> line(width * static_cast<size_t>(components));
  d.run([&](jpeg_decompress_struct& cinfo) {
    while (cinfo.output_scanline < cinfo.output_height) {
      uint8_t* row = base + (stride * cinfo.output_scanline);
      JSAMPROW in = line.data();
      jpeg_read_scanlines(&cinfo, &in, 1);
      convert_row(line.data(), components, row, kept, width);
    }
    jpeg_finish_decompress(&cinfo);
  });
  return pixels;
}

}  // namespace torchrkt::detail
