#include <gtest/gtest.h>

#include <cstdint>
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

std::vector<float> cpu_data_of(const tr_tensor* t) {
  std::uint64_t numel = 0;
  EXPECT_EQ(tr_tensor_copy_data(t, 0, nullptr, &numel), 2) << tr_last_error();
  std::vector<float> out(numel);
  EXPECT_EQ(tr_tensor_copy_data(t, numel, out.data(), &numel), 0)
      << tr_last_error();
  return out;
}

struct DefaultDeviceGuard {
  DefaultDeviceGuard() = default;
  DefaultDeviceGuard(const DefaultDeviceGuard&) = delete;
  DefaultDeviceGuard& operator=(const DefaultDeviceGuard&) = delete;
  ~DefaultDeviceGuard() {
    tr_set_default_device(TR_DEVICE_CPU, 0);
  }
};

TEST(TorchrktDevice, DefaultsToCpu) {
  // reset: --gtest_shuffle could run a CUDA test's abort before its guard
  tr_set_default_device(TR_DEVICE_CPU, 0);
  tr_device_type type = TR_DEVICE_CUDA;
  int64_t index = -1;
  EXPECT_EQ(tr_get_default_device(&type, &index), 0) << tr_last_error();
  EXPECT_EQ(type, TR_DEVICE_CPU);
  EXPECT_EQ(index, 0);
}

TEST(TorchrktDevice, NewTensorsLandOnDefaultDevice) {
  const std::vector<int64_t> dims = {2, 2};
  const Handle t(tr_zeros(dims.data(), 2));
  tr_device_type type = TR_DEVICE_CUDA;
  int64_t index = -1;
  EXPECT_EQ(tr_tensor_device(t.t, &type, &index), 0) << tr_last_error();
  EXPECT_EQ(type, TR_DEVICE_CPU);
}

TEST(TorchrktDevice, ToDeviceCpuIsIdentity) {
  const std::vector<float> values = {1.0F, 2.0F, 3.0F};
  const std::vector<int64_t> dims = {3};
  const Handle src(tr_from_data(values.data(), values.size(), dims.data(), 1));
  const Handle moved(tr_tensor_to_device(src.t, TR_DEVICE_CPU, 0));
  EXPECT_EQ(cpu_data_of(moved.t), values);
}

TEST(TorchrktDevice, FromDataOnPlacesOnExplicitCpu) {
  const std::vector<float> values = {1.0F, 2.0F, 3.0F};
  const std::vector<int64_t> dims = {3};
  const Handle t(tr_from_data_on_device(values.data(), values.size(),
                                        dims.data(), 1, TR_DEVICE_CPU, 0));
  ASSERT_NE(t.t, nullptr) << tr_last_error();
  tr_device_type type = TR_DEVICE_CUDA;
  int64_t index = -1;
  EXPECT_EQ(tr_tensor_device(t.t, &type, &index), 0) << tr_last_error();
  EXPECT_EQ(type, TR_DEVICE_CPU);
  EXPECT_EQ(cpu_data_of(t.t), values);
}

TEST(TorchrktDevice, FromDataOnRejectsOutOfRangeCudaIndex) {
  // 256 would wrap torch's 8-bit DeviceIndex to device 0 if unvalidated
  const std::vector<float> values = {1.0F};
  const std::vector<int64_t> dims = {1};
  EXPECT_EQ(tr_from_data_on_device(values.data(), values.size(), dims.data(), 1,
                                   TR_DEVICE_CUDA, 256),
            nullptr);
  EXPECT_STRNE(tr_last_error(), "");
}

TEST(TorchrktDevice, FromDataOnRejectsUnknownDeviceType) {
  const std::vector<float> values = {1.0F};
  const std::vector<int64_t> dims = {1};
  EXPECT_EQ(tr_from_data_on_device(values.data(), values.size(), dims.data(), 1,
                                   static_cast<tr_device_type>(99), 0),
            nullptr);
  EXPECT_STRNE(tr_last_error(), "");
}

TEST(TorchrktDevice, CudaMemoryStatsFailsCleanlyWithoutCuda) {
  if (tr_cuda_is_available() != 0) {
    GTEST_SKIP() << "CUDA present; the success path is CudaRoundTrip";
  }
  int64_t alloc = -1;
  int64_t reserved = -1;
  int64_t peak = -1;
  EXPECT_EQ(tr_cuda_memory_stats(0, &alloc, &reserved, &peak), 1);
  EXPECT_STRNE(tr_last_error(), "");
  EXPECT_EQ(tr_cuda_memory_stats(0, nullptr, nullptr, nullptr), 1);
}

TEST(TorchrktDevice, EmptyCacheIsNoOpSuccessWithoutCuda) {
  if (tr_cuda_is_available() != 0) {
    GTEST_SKIP() << "CUDA present; the success path is CudaRoundTrip";
  }
  EXPECT_EQ(tr_cuda_empty_cache(), 0);
}

TEST(TorchrktDevice, NullArgsReportStatus) {
  EXPECT_EQ(tr_tensor_to_device(nullptr, TR_DEVICE_CPU, 0), nullptr);
  EXPECT_STRNE(tr_last_error(), "");
  EXPECT_EQ(tr_tensor_device(nullptr, nullptr, nullptr), 1);
  EXPECT_STRNE(tr_last_error(), "");
  EXPECT_EQ(tr_get_default_device(nullptr, nullptr), 1);
  EXPECT_STRNE(tr_last_error(), "");
}

TEST(TorchrktDevice, SetCpuNonzeroIndexErrors) {
  const DefaultDeviceGuard guard;
  EXPECT_EQ(tr_set_default_device(TR_DEVICE_CPU, 1), 1);
  EXPECT_STRNE(tr_last_error(), "")
      << "expected an error for a non-zero CPU index";
  tr_device_type type = TR_DEVICE_CUDA;
  int64_t index = -1;
  EXPECT_EQ(tr_get_default_device(&type, &index), 0) << tr_last_error();
  EXPECT_EQ(type, TR_DEVICE_CPU);
  EXPECT_EQ(index, 0);
}

TEST(TorchrktDevice, SetCudaDefaultWhenUnavailableErrors) {
  if (tr_cuda_is_available() != 0) {
    GTEST_SKIP() << "CUDA present; the success path is CudaRoundTrip";
  }
  const DefaultDeviceGuard guard;
  EXPECT_EQ(tr_set_default_device(TR_DEVICE_CUDA, 0), 1);
  EXPECT_STRNE(tr_last_error(), "") << "expected an error after a failed set";
  tr_device_type type = TR_DEVICE_CUDA;
  int64_t index = -1;
  EXPECT_EQ(tr_get_default_device(&type, &index), 0) << tr_last_error();
  EXPECT_EQ(type, TR_DEVICE_CPU);
}

TEST(TorchrktDevice, CudaRoundTrip) {
  if (tr_cuda_is_available() == 0) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  EXPECT_GT(tr_cuda_device_count(), 0);
  const DefaultDeviceGuard guard;
  ASSERT_EQ(tr_set_default_device(TR_DEVICE_CUDA, 0), 0) << tr_last_error();

  const std::vector<int64_t> dims = {2, 2};
  const Handle on_gpu(tr_zeros(dims.data(), 2));
  tr_device_type type = TR_DEVICE_CPU;
  int64_t index = -1;
  EXPECT_EQ(tr_tensor_device(on_gpu.t, &type, &index), 0) << tr_last_error();
  EXPECT_EQ(type, TR_DEVICE_CUDA);

  const Handle back(tr_tensor_to_device(on_gpu.t, TR_DEVICE_CPU, 0));
  EXPECT_EQ(cpu_data_of(back.t), (std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));

  const std::vector<float> host_vals = {1.0F, 2.0F, 3.0F};
  const std::vector<int64_t> src_dims = {3};
  const Handle from_data_gpu(
      tr_from_data(host_vals.data(), host_vals.size(), src_dims.data(), 1));
  tr_device_type fd_type = TR_DEVICE_CPU;
  int64_t fd_index = -1;
  EXPECT_EQ(tr_tensor_device(from_data_gpu.t, &fd_type, &fd_index), 0)
      << tr_last_error();
  EXPECT_EQ(fd_type, TR_DEVICE_CUDA);
  const Handle fd_back(tr_tensor_to_device(from_data_gpu.t, TR_DEVICE_CPU, 0));
  EXPECT_EQ(cpu_data_of(fd_back.t), host_vals);

  const Handle explicit_cpu(
      tr_from_data_on_device(host_vals.data(), host_vals.size(),
                             src_dims.data(), 1, TR_DEVICE_CPU, 0));
  tr_device_type ec_type = TR_DEVICE_CUDA;
  int64_t ec_index = -1;
  EXPECT_EQ(tr_tensor_device(explicit_cpu.t, &ec_type, &ec_index), 0)
      << tr_last_error();
  EXPECT_EQ(ec_type, TR_DEVICE_CPU);
  EXPECT_EQ(cpu_data_of(explicit_cpu.t), host_vals);

  const Handle on_cuda(tr_from_data_on_device(host_vals.data(),
                                              host_vals.size(), src_dims.data(),
                                              1, TR_DEVICE_CUDA, 0));
  tr_device_type oc_type = TR_DEVICE_CPU;
  int64_t oc_index = -1;
  EXPECT_EQ(tr_tensor_device(on_cuda.t, &oc_type, &oc_index), 0)
      << tr_last_error();
  EXPECT_EQ(oc_type, TR_DEVICE_CUDA);
  const Handle oc_back(tr_tensor_to_device(on_cuda.t, TR_DEVICE_CPU, 0));
  EXPECT_EQ(cpu_data_of(oc_back.t), host_vals);

  int64_t alloc = -1;
  int64_t reserved = -1;
  int64_t peak = -1;
  ASSERT_EQ(tr_cuda_memory_stats(0, &alloc, &reserved, &peak), 0)
      << tr_last_error();
  EXPECT_GT(alloc, 0);
  EXPECT_GE(peak, alloc);
  EXPECT_GE(reserved, alloc);
  EXPECT_EQ(tr_cuda_empty_cache(), 0) << tr_last_error();
  int64_t reserved_after = -1;
  ASSERT_EQ(tr_cuda_memory_stats(0, &alloc, &reserved_after, &peak), 0)
      << tr_last_error();
  EXPECT_LE(reserved_after, reserved);
  EXPECT_EQ(tr_cuda_memory_stats(256, &alloc, &reserved, &peak), 1);
  EXPECT_STRNE(tr_last_error(), "");

  const std::vector<int64_t> randn_dims = {4};
  const Handle randn_gpu(tr_randn(randn_dims.data(), 1));
  tr_device_type rd_type = TR_DEVICE_CPU;
  int64_t rd_index = -1;
  EXPECT_EQ(tr_tensor_device(randn_gpu.t, &rd_type, &rd_index), 0)
      << tr_last_error();
  EXPECT_EQ(rd_type, TR_DEVICE_CUDA);
}

}  // namespace
