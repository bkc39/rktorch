#include <ATen/Context.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "torchrkt/c_api.h"

namespace {

using at::Float32Backend;
using at::Float32MatmulPrecision;
using at::Float32Op;
using at::Float32Precision;

const char* precision_name(Float32MatmulPrecision p) {
  switch (p) {
    case Float32MatmulPrecision::HIGH:
      return "high";
    case Float32MatmulPrecision::MEDIUM:
      return "medium";
    case Float32MatmulPrecision::HIGHEST:
      break;
  }
  return "highest";
}

class SwitchesRestored {
 public:
  SwitchesRestored()
      : algorithms_(at::globalContext().deterministicAlgorithms()),
        warn_only_(at::globalContext().deterministicAlgorithmsWarnOnly()),
        cudnn_deterministic_(at::globalContext().deterministicCuDNN()),
        cudnn_benchmark_(at::globalContext().benchmarkCuDNN()),
        precision_(at::globalContext().float32MatmulPrecision()),
        cudnn_tf32_(at::globalContext().allowTF32CuDNN()) {}
  SwitchesRestored(const SwitchesRestored&) = delete;
  SwitchesRestored& operator=(const SwitchesRestored&) = delete;
  SwitchesRestored(SwitchesRestored&&) = delete;
  SwitchesRestored& operator=(SwitchesRestored&&) = delete;
  ~SwitchesRestored() {
    at::globalContext().setDeterministicAlgorithms(algorithms_, warn_only_);
    at::globalContext().setDeterministicCuDNN(cudnn_deterministic_);
    at::globalContext().setBenchmarkCuDNN(cudnn_benchmark_);
    at::globalContext().setFloat32MatmulPrecision(precision_name(precision_));
    at::globalContext().setAllowTF32CuDNN(cudnn_tf32_);
  }

 private:
  bool algorithms_;
  bool warn_only_;
  bool cudnn_deterministic_;
  bool cudnn_benchmark_;
  Float32MatmulPrecision precision_;
  bool cudnn_tf32_;
};

class CapturingHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    messages.push_back(warning.msg());
  }
  std::vector<std::string> messages;
};

int read_flag(int (*get)(int*)) {
  int out = -1;
  EXPECT_EQ(get(&out), 0) << tr_last_error();
  return out;
}

struct Alert {
  bool threw = false;
  std::string error;
  std::vector<std::string> warnings;
};

Alert probe_alert() {
  CapturingHandler handler;
  Alert alert;
  {
    c10::WarningUtils::WarningHandlerGuard guard(&handler);
    try {
      at::globalContext().alertNotDeterministic("probe");
    } catch (const c10::Error& e) {
      alert.threw = true;
      alert.error = e.what();
    }
  }
  alert.warnings = handler.messages;
  return alert;
}

TEST(TorchrktSwitches, SettersWriteTheContextAndGettersReadIt) {
  SwitchesRestored restore;
  at::Context& ctx = at::globalContext();

  ASSERT_EQ(tr_set_deterministic_algorithms(1, 0), 0) << tr_last_error();
  EXPECT_TRUE(ctx.deterministicAlgorithms());
  EXPECT_FALSE(ctx.deterministicAlgorithmsWarnOnly());
  ASSERT_EQ(tr_set_deterministic_algorithms(0, 1), 0) << tr_last_error();
  EXPECT_FALSE(ctx.deterministicAlgorithms());
  EXPECT_TRUE(ctx.deterministicAlgorithmsWarnOnly());

  ASSERT_EQ(tr_set_cudnn_deterministic(1), 0) << tr_last_error();
  EXPECT_TRUE(ctx.deterministicCuDNN());
  EXPECT_FALSE(ctx.benchmarkCuDNN());
  ASSERT_EQ(tr_set_cudnn_deterministic(0), 0) << tr_last_error();
  EXPECT_FALSE(ctx.deterministicCuDNN());

  ASSERT_EQ(tr_set_cudnn_benchmark(1), 0) << tr_last_error();
  EXPECT_TRUE(ctx.benchmarkCuDNN());
  EXPECT_FALSE(ctx.deterministicCuDNN());
  ASSERT_EQ(tr_set_cudnn_benchmark(0), 0) << tr_last_error();
  EXPECT_FALSE(ctx.benchmarkCuDNN());

  ctx.setDeterministicAlgorithms(true, false);
  EXPECT_EQ(read_flag(tr_get_deterministic_algorithms), 1);
  EXPECT_EQ(read_flag(tr_get_deterministic_algorithms_warn_only), 0);
  ctx.setDeterministicAlgorithms(false, true);
  EXPECT_EQ(read_flag(tr_get_deterministic_algorithms), 0);
  EXPECT_EQ(read_flag(tr_get_deterministic_algorithms_warn_only), 1);

  ctx.setDeterministicCuDNN(true);
  EXPECT_EQ(read_flag(tr_get_cudnn_deterministic), 1);
  EXPECT_EQ(read_flag(tr_get_cudnn_benchmark), 0);
  ctx.setDeterministicCuDNN(false);
  EXPECT_EQ(read_flag(tr_get_cudnn_deterministic), 0);

  ctx.setBenchmarkCuDNN(true);
  EXPECT_EQ(read_flag(tr_get_cudnn_benchmark), 1);
  EXPECT_EQ(read_flag(tr_get_cudnn_deterministic), 0);
  ctx.setBenchmarkCuDNN(false);
  EXPECT_EQ(read_flag(tr_get_cudnn_benchmark), 0);
}

TEST(TorchrktSwitches, TheSwitchRefusesOrWarnsAtTheAlert) {
  SwitchesRestored restore;

  ASSERT_EQ(tr_set_deterministic_algorithms(1, 0), 0) << tr_last_error();
  const Alert strict = probe_alert();
  EXPECT_TRUE(strict.threw);
  EXPECT_NE(
      strict.error.find("probe does not have a deterministic implementation"),
      std::string::npos)
      << strict.error;
  EXPECT_TRUE(strict.warnings.empty());

  ASSERT_EQ(tr_set_deterministic_algorithms(1, 1), 0) << tr_last_error();
  const Alert warned = probe_alert();
  EXPECT_FALSE(warned.threw) << warned.error;
  ASSERT_EQ(warned.warnings.size(), 1U);
  EXPECT_NE(warned.warnings[0].find("warn_only=True"), std::string::npos)
      << warned.warnings[0];

  for (const int warn_only : {0, 1}) {
    ASSERT_EQ(tr_set_deterministic_algorithms(0, warn_only), 0)
        << tr_last_error();
    const Alert off = probe_alert();
    EXPECT_FALSE(off.threw) << off.error;
    EXPECT_TRUE(off.warnings.empty()) << "warn_only " << warn_only;
  }
}

TEST(TorchrktSwitches, TheSwitchFillsNewMemory) {
  SwitchesRestored restore;
  ASSERT_EQ(tr_set_deterministic_algorithms(1, 0), 0) << tr_last_error();

  const torch::Tensor floats = torch::empty({3});
  for (int64_t i = 0; i < floats.numel(); ++i) {
    EXPECT_TRUE(std::isnan(floats[i].item<float>())) << "element " << i;
  }
  const torch::Tensor longs = torch::empty({2}, torch::kLong);
  for (int64_t i = 0; i < longs.numel(); ++i) {
    EXPECT_EQ(longs[i].item<int64_t>(), std::numeric_limits<int64_t>::max())
        << "element " << i;
  }
}

TEST(TorchrktSwitches, GettersRefuseANullOut) {
  const std::vector<std::pair<const char*, int (*)(int*)>> getters = {
      {"tr_get_deterministic_algorithms", tr_get_deterministic_algorithms},
      {"tr_get_deterministic_algorithms_warn_only",
       tr_get_deterministic_algorithms_warn_only},
      {"tr_get_cudnn_deterministic", tr_get_cudnn_deterministic},
      {"tr_get_cudnn_benchmark", tr_get_cudnn_benchmark},
      {"tr_get_float32_matmul_precision", tr_get_float32_matmul_precision},
      {"tr_get_cuda_matmul_allow_tf32", tr_get_cuda_matmul_allow_tf32},
      {"tr_get_cudnn_allow_tf32", tr_get_cudnn_allow_tf32},
  };
  for (const auto& [name, get] : getters) {
    EXPECT_EQ(get(nullptr), 1) << name;
    EXPECT_EQ(std::string(tr_last_error()).rfind(std::string(name) + ":", 0),
              0U)
        << name << ": " << tr_last_error();
  }
}

Float32Precision onednn_matmul() {
  return at::globalContext().float32Precision(Float32Backend::MKLDNN,
                                              Float32Op::MATMUL);
}

Float32Precision cuda_matmul() {
  return at::globalContext().float32Precision(Float32Backend::CUDA,
                                              Float32Op::MATMUL);
}

TEST(TorchrktSwitches, PrecisionSettersWriteTheContext) {
  SwitchesRestored restore;
  at::Context& ctx = at::globalContext();
  const std::vector<Float32Precision> onednn = {
      Float32Precision::IEEE, Float32Precision::TF32, Float32Precision::BF16};

  for (const bool cudnn : {true, false}) {
    ctx.setAllowTF32CuDNN(cudnn);
    for (int code = 0; code < 3; ++code) {
      ASSERT_EQ(tr_set_float32_matmul_precision(code), 0) << tr_last_error();
      EXPECT_EQ(ctx.float32MatmulPrecision(),
                static_cast<Float32MatmulPrecision>(code));
      EXPECT_EQ(ctx.allowTF32CuBLAS(), code != 0) << "code " << code;
      EXPECT_EQ(cuda_matmul(),
                code != 0 ? Float32Precision::TF32 : Float32Precision::IEEE)
          << "code " << code;
      EXPECT_EQ(onednn_matmul(), onednn[code]) << "code " << code;
      EXPECT_EQ(ctx.allowTF32CuDNN(), cudnn) << "code " << code;
    }
  }

  for (const bool on : {true, false}) {
    ctx.setFloat32MatmulPrecision("highest");
    ASSERT_EQ(tr_set_cuda_matmul_allow_tf32(on ? 1 : 0), 0) << tr_last_error();
    EXPECT_EQ(ctx.allowTF32CuBLAS(), on);
    EXPECT_EQ(
        ctx.float32MatmulPrecision(),
        on ? Float32MatmulPrecision::HIGH : Float32MatmulPrecision::HIGHEST);
    EXPECT_EQ(onednn_matmul(), Float32Precision::IEEE) << "on " << on;
  }

  ctx.setFloat32MatmulPrecision("high");
  for (const bool on : {false, true}) {
    ASSERT_EQ(tr_set_cudnn_allow_tf32(on ? 1 : 0), 0) << tr_last_error();
    EXPECT_EQ(ctx.allowTF32CuDNN(), on);
    EXPECT_EQ(ctx.allowTF32CuDNN(Float32Op::CONV), on);
    EXPECT_EQ(ctx.allowTF32CuDNN(Float32Op::RNN), on);
    EXPECT_EQ(ctx.float32MatmulPrecision(), Float32MatmulPrecision::HIGH);
    EXPECT_TRUE(ctx.allowTF32CuBLAS());
    EXPECT_EQ(onednn_matmul(), Float32Precision::TF32);
  }
}

TEST(TorchrktSwitches, PrecisionGettersReadTheContext) {
  SwitchesRestored restore;
  at::Context& ctx = at::globalContext();
  const std::vector<std::pair<const char*, int>> names = {
      {"medium", 2}, {"high", 1}, {"highest", 0}};
  for (const auto& [name, code] : names) {
    ctx.setFloat32MatmulPrecision(name);
    EXPECT_EQ(read_flag(tr_get_float32_matmul_precision), code) << name;
    EXPECT_EQ(read_flag(tr_get_cuda_matmul_allow_tf32), code != 0) << name;
  }
  for (const bool on : {true, false}) {
    ctx.setAllowTF32CuBLAS(on);
    EXPECT_EQ(read_flag(tr_get_cuda_matmul_allow_tf32), on ? 1 : 0);
    EXPECT_EQ(read_flag(tr_get_float32_matmul_precision), on ? 1 : 0);
    ctx.setAllowTF32CuDNN(on);
    EXPECT_EQ(read_flag(tr_get_cudnn_allow_tf32), on ? 1 : 0);
  }
}

TEST(TorchrktSwitches, PrecisionSetterRefusesAnUnknownCode) {
  SwitchesRestored restore;
  at::Context& ctx = at::globalContext();
  ctx.setFloat32MatmulPrecision("high");
  for (const int code : {3, -1}) {
    CapturingHandler handler;
    int rc = 0;
    {
      c10::WarningUtils::WarningHandlerGuard guard(&handler);
      rc = tr_set_float32_matmul_precision(code);
    }
    EXPECT_EQ(rc, 1) << "code " << code;
    EXPECT_EQ(std::string(tr_last_error())
                  .rfind("tr_set_float32_matmul_precision:", 0),
              0U)
        << tr_last_error();
    EXPECT_EQ(ctx.float32MatmulPrecision(), Float32MatmulPrecision::HIGH);
    EXPECT_EQ(onednn_matmul(), Float32Precision::TF32);
    EXPECT_TRUE(handler.messages.empty()) << handler.messages.front();
  }
}

TEST(TorchrktSwitches, PrecisionGetterRefusesAMixedState) {
  SwitchesRestored restore;
  ASSERT_EQ(tr_set_float32_matmul_precision(1), 0) << tr_last_error();
  ASSERT_EQ(tr_set_cuda_matmul_allow_tf32(0), 0) << tr_last_error();
  int out = -7;
  EXPECT_EQ(tr_get_float32_matmul_precision(&out), 1);
  EXPECT_EQ(out, -7);
  EXPECT_NE(std::string(tr_last_error()).find("mix of the legacy and new APIs"),
            std::string::npos)
      << tr_last_error();
  out = -7;
  EXPECT_EQ(tr_get_cuda_matmul_allow_tf32(&out), 0) << tr_last_error();
  EXPECT_EQ(out, 0);
  ASSERT_EQ(tr_set_float32_matmul_precision(0), 0) << tr_last_error();
  EXPECT_EQ(read_flag(tr_get_float32_matmul_precision), 0);
}

}  // namespace
