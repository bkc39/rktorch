// White-box like finalizer_death_test: exercises the shared exception
// classifier directly, since the high-water-mark MPS refusal cannot be
// produced by a real allocation without RAM-scale commits.
#include <gtest/gtest.h>

#include <new>
#include <stdexcept>

#include "torchrkt/detail/op_call.hpp"

namespace {

using torchrkt::classify;
using torchrkt::error_kind;

TEST(TorchrktClassify, AllocatorMessageShapesClassifyAsOom) {
  EXPECT_EQ(classify(std::runtime_error(
                "DefaultCPUAllocator: not enough memory: you tried to "
                "allocate 4611686018427387904 bytes.")),
            error_kind::oom);
  EXPECT_EQ(classify(std::runtime_error(
                "MPS backend out of memory (MPS allocated: 1.00 GB, other "
                "allocations: 0 bytes, max allowed: 1.70 GB).")),
            error_kind::oom);
  EXPECT_EQ(classify(std::runtime_error("Invalid buffer size: 4096.00 GiB")),
            error_kind::oom);
}

TEST(TorchrktClassify, TypedBadAllocClassifiesAsOom) {
  EXPECT_EQ(classify(std::bad_alloc()), error_kind::oom);
}

TEST(TorchrktClassify, OrdinaryErrorsStayGeneric) {
  EXPECT_EQ(classify(std::runtime_error("shape mismatch: [2] vs [3]")),
            error_kind::generic);
  EXPECT_EQ(classify(std::runtime_error("buffer size invalid")),
            error_kind::generic);
}

}  // namespace
