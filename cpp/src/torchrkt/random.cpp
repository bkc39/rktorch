#include "torchrkt/c_api/random.h"

#include <torch/torch.h>

#include <exception>
#include <string>
#include <vector>

#include "torchrkt/detail/error.hpp"
#include "torchrkt/detail/tensor_handle.hpp"

extern "C" {

tr_tensor* tr_randn(const int64_t* dims, int64_t ndim) {
  if (ndim < 0 || (ndim > 0 && !dims)) {
    torchrkt::set_error("tr_randn: ndim/dims inconsistent");
    return nullptr;
  }
  try {
    const std::vector<int64_t> shape(dims, dims + ndim);
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return new tr_tensor{torch::randn(shape, options)};
  } catch (const std::exception& e) {
    torchrkt::set_error(std::string("tr_randn: ") + e.what());
    return nullptr;
  } catch (...) {
    torchrkt::set_error("tr_randn: unknown exception");
    return nullptr;
  }
}

}  // extern "C"
