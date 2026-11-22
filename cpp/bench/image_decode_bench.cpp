// The decoders alone against tr_image_decode, one thread, HWC uint8 RGB out:
//
//   image_decode_bench [--reps N] FILE...
//
// prints, per arm, the median, min and max over N passes of the mean
// microseconds per image, then how far each arm's pixels are from the first
// arm's. Not built by default (-DTORCHRKT_BUILD_BENCH=ON) and never a gate.

// jpeglib.h uses FILE and size_t without including their headers
// clang-format off
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>
// clang-format on
#include <stb_image.h>
#include <turbojpeg.h>

#ifdef TORCHRKT_BENCH_SPNG
#include <spng.h>
#endif
#ifdef TORCHRKT_BENCH_LIBPNG
#include <png.h>
#endif

#include <algorithm>
#include <chrono>
#include <csetjmp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "torchrkt/c_api.h"

namespace {

using Bytes = std::vector<uint8_t>;
using Decoder = std::function<Bytes(const Bytes&)>;

Bytes read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

Bytes stb(const Bytes& data) {
  int w = 0;
  int h = 0;
  int n = 0;
  stbi_uc* px = stbi_load_from_memory(
      data.data(), static_cast<int>(data.size()), &w, &h, &n, 3);
  if (px == nullptr) {
    throw std::runtime_error(stbi_failure_reason());
  }
  Bytes out(px, px + static_cast<size_t>(w) * h * 3);
  stbi_image_free(px);
  return out;
}

Bytes turbo(const Bytes& data) {
  thread_local tjhandle tj = tj3Init(TJINIT_DECOMPRESS);
  if (tj3DecompressHeader(tj, data.data(), data.size()) != 0) {
    throw std::runtime_error(tj3GetErrorStr(tj));
  }
  const int w = tj3Get(tj, TJPARAM_JPEGWIDTH);
  const int h = tj3Get(tj, TJPARAM_JPEGHEIGHT);
  Bytes out(static_cast<size_t>(w) * h * 3);
  if (tj3Decompress8(tj, data.data(), data.size(), out.data(), 0, TJPF_RGB) !=
      0) {
    throw std::runtime_error(tj3GetErrorStr(tj));
  }
  return out;
}

struct JpegError {
  jpeg_error_mgr pub;
  std::jmp_buf jump;
};

void jpeg_bail(j_common_ptr cinfo) {
  std::longjmp(reinterpret_cast<JpegError*>(cinfo->err)->jump, 1);
}

// torchvision's decode_jpeg: libjpeg's defaults, one scanline per call
Bytes libjpeg(const Bytes& data) {
  jpeg_decompress_struct cinfo{};
  JpegError err{};
  cinfo.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = jpeg_bail;
  Bytes out;
  if (setjmp(err.jump) != 0) {
    jpeg_destroy_decompress(&cinfo);
    throw std::runtime_error("libjpeg failed");
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, data.data(), data.size());
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);
  const size_t stride = static_cast<size_t>(cinfo.output_width) * 3;
  out.resize(stride * cinfo.output_height);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row = out.data() + stride * cinfo.output_scanline;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return out;
}

Bytes shim(const Bytes& data) {
  tr_tensor* t =
      tr_image_decode(data.data(), static_cast<int64_t>(data.size()), 3);
  if (t == nullptr) {
    throw std::runtime_error(tr_last_error());
  }
  int64_t numel = 0;
  tr_tensor_numel(t, &numel);
  const std::vector<int64_t> dims = {1, 2, 0};
  tr_tensor* hwc = tr_permute(t, dims.data(), 3);
  Bytes out(static_cast<size_t>(numel));
  uint64_t got = 0;
  tr_tensor_copy_data_u8(hwc, out.size(), out.data(), &got);
  tr_tensor_free(hwc);
  tr_tensor_free(t);
  return out;
}

// the shim's own work only: the decode, its tensor, and the free
Bytes shim_call(const Bytes& data) {
  tr_tensor* t =
      tr_image_decode(data.data(), static_cast<int64_t>(data.size()), 3);
  if (t == nullptr) {
    throw std::runtime_error(tr_last_error());
  }
  tr_tensor_free(t);
  return {};
}

#ifdef TORCHRKT_BENCH_SPNG
Bytes spng(const Bytes& data) {
  spng_ctx* ctx = spng_ctx_new(0);
  spng_set_png_buffer(ctx, data.data(), data.size());
  size_t size = 0;
  spng_decoded_image_size(ctx, SPNG_FMT_RGB8, &size);
  Bytes out(size);
  const int rc = spng_decode_image(ctx, out.data(), size, SPNG_FMT_RGB8, 0);
  spng_ctx_free(ctx);
  if (rc != 0) {
    throw std::runtime_error(spng_strerror(rc));
  }
  return out;
}
#endif

#ifdef TORCHRKT_BENCH_LIBPNG
Bytes libpng(const Bytes& data) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_memory(&image, data.data(), data.size()) == 0) {
    throw std::runtime_error(image.message);
  }
  image.format = PNG_FORMAT_RGB;
  Bytes out(PNG_IMAGE_SIZE(image));
  if (png_image_finish_read(&image, nullptr, out.data(), 0, nullptr) == 0) {
    throw std::runtime_error(image.message);
  }
  return out;
}
#endif

bool is_jpeg(const Bytes& data) {
  return data.size() >= 3 && data[0] == 0xff && data[1] == 0xd8 &&
         data[2] == 0xff;
}

std::vector<std::pair<std::string, Decoder>> arms(bool jpeg) {
  if (jpeg) {
    return {{"stb", stb},
            {"turbo-tj3", turbo},
            {"libjpeg-scanline", libjpeg},
            {"tr_image_decode", shim},
            {"tr_image_decode-call", shim_call}};
  }
  return {{"stb", stb},
#ifdef TORCHRKT_BENCH_SPNG
          {"spng", spng},
#endif
#ifdef TORCHRKT_BENCH_LIBPNG
          {"libpng", libpng},
#endif
          {"tr_image_decode", shim},
          {"tr_image_decode-call", shim_call}};
}

double pass_us(const Decoder& f, const std::vector<Bytes>& files) {
  const auto t0 = std::chrono::steady_clock::now();
  for (const Bytes& data : files) {
    f(data);
  }
  const std::chrono::duration<double, std::micro> dt =
      std::chrono::steady_clock::now() - t0;
  return dt.count() / static_cast<double>(files.size());
}

int max_abs_diff(const Bytes& a, const Bytes& b) {
  if (a.size() != b.size()) {
    return -1;
  }
  int worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(int{a[i]} - int{b[i]}));
  }
  return worst;
}

void run(const std::string& kind, const std::vector<Bytes>& files, int reps) {
  if (files.empty()) {
    return;
  }
  const auto named = arms(kind == "jpeg");
  std::printf("%s: %zu files, %d passes\n", kind.c_str(), files.size(), reps);
  std::printf("  %-22s %10s %10s %10s\n", "arm", "median_us", "min_us",
              "max_us");
  for (const auto& [name, f] : named) {
    pass_us(f, files);
    std::vector<double> passes;
    for (int r = 0; r < reps; ++r) {
      passes.push_back(pass_us(f, files));
    }
    std::sort(passes.begin(), passes.end());
    std::printf("  %-22s %10.1f %10.1f %10.1f\n", name.c_str(),
                passes[passes.size() / 2], passes.front(), passes.back());
  }
  for (const auto& [name, f] : named) {
    if (name == named.front().first || name == "tr_image_decode-call") {
      continue;
    }
    int worst = 0;
    for (const Bytes& data : files) {
      worst =
          std::max(worst, max_abs_diff(named.front().second(data), f(data)));
    }
    std::printf("  max |%s - %s| = %d\n", name.c_str(),
                named.front().first.c_str(), worst);
  }
}

}  // namespace

int main(int argc, char** argv) {
  int reps = 5;
  std::vector<Bytes> jpegs;
  std::vector<Bytes> pngs;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--reps" && i + 1 < argc) {
      reps = std::max(1, std::atoi(argv[++i]));
      continue;
    }
    Bytes data = read_file(arg);
    (is_jpeg(data) ? jpegs : pngs).push_back(std::move(data));
  }
  run("jpeg", jpegs, reps);
  run("png", pngs, reps);
  return 0;
}
