#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "torchrkt/c_api.h"

namespace {

constexpr int kThreads = 8;
constexpr int kRounds = 200;
constexpr int64_t kSide = 16;

struct Handle {
  tr_tensor* t;
  explicit Handle(tr_tensor* p) : t(p) {}
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept : t(other.t) {
    other.t = nullptr;
  }
  Handle& operator=(Handle&&) = delete;
  ~Handle() {
    tr_tensor_free(t);
  }
};

float scalar_of(const tr_tensor* t) {
  std::uint64_t numel = 1;
  float out = 0.0F;
  if (tr_tensor_copy_data(t, 1, &out, &numel) != 0) {
    return -1.0F;
  }
  return out;
}

Handle square(tr_tensor* (*make)(const int64_t*, int64_t)) {
  const std::vector<int64_t> dims = {kSide, kSide};
  return Handle(make(dims.data(), 2));
}

float one_round() {
  const Handle ones = square(tr_ones);
  const Handle twos = Handle(tr_add(ones.t, ones.t));
  const Handle product = Handle(tr_matmul(twos.t, ones.t));
  const Handle total = Handle(tr_sum(product.t));
  return scalar_of(total.t);
}

template <typename Body>
void run_threads(int count, Body body) {
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(count));
  for (int k = 0; k < count; ++k) {
    threads.emplace_back(body, k);
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(TorchrktThreads, ThreadsCreateOperateAndFreeIndependently) {
  constexpr float expected = 2.0F * kSide * kSide * kSide;
  std::atomic<int> wrong{0};
  run_threads(kThreads, [&wrong](int /*k*/) {
    for (int i = 0; i < kRounds; ++i) {
      if (one_round() != expected) {
        wrong.fetch_add(1);
      }
    }
  });
  EXPECT_EQ(wrong.load(), 0);
}

TEST(TorchrktThreads, ViewsOfOneStorageFreeInAnyOrderOnAnyThread) {
  const std::vector<int64_t> dims = {kThreads, kSide};
  std::vector<Handle> views;
  views.reserve(kThreads);
  {
    const Handle base = Handle(tr_ones(dims.data(), 2));
    for (int k = 0; k < kThreads; ++k) {
      views.emplace_back(tr_gen_narrow(base.t, 0, k, 1));
    }
  }
  std::atomic<int> wrong{0};
  run_threads(kThreads, [&views, &wrong](int k) {
    const Handle total = Handle(tr_sum(views[static_cast<size_t>(k)].t));
    if (scalar_of(total.t) != static_cast<float>(kSide)) {
      wrong.fetch_add(1);
    }
    tr_tensor_free(views[static_cast<size_t>(k)].t);
    views[static_cast<size_t>(k)].t = nullptr;
  });
  EXPECT_EQ(wrong.load(), 0);
}

std::string failing_mm(int64_t inner) {
  const std::vector<int64_t> left = {2, inner};
  const std::vector<int64_t> right = {inner + 1, 2};
  const Handle a = Handle(tr_zeros(left.data(), 2));
  const Handle b = Handle(tr_zeros(right.data(), 2));
  const Handle product = Handle(tr_mm(a.t, b.t));
  return product.t == nullptr ? std::string("failed") : std::string();
}

TEST(TorchrktThreads, LastErrorBelongsToTheFailingThread) {
  std::barrier both_failed(2);
  std::vector<std::string> messages(2);
  run_threads(2, [&both_failed, &messages](int k) {
    const int64_t inner = k == 0 ? 3 : 5;
    const std::string outcome = failing_mm(inner);
    both_failed.arrive_and_wait();
    messages[static_cast<size_t>(k)] = outcome + ": " + tr_last_error();
  });
  EXPECT_NE(messages[0].find("2x3 and 4x2"), std::string::npos) << messages[0];
  EXPECT_NE(messages[1].find("2x5 and 6x2"), std::string::npos) << messages[1];
}

TEST(TorchrktThreads, GradModeIsPerThread) {
  std::barrier set(2);
  std::vector<int> seen(2, -1);
  run_threads(2, [&set, &seen](int k) {
    if (k == 0) {
      EXPECT_EQ(tr_set_grad_enabled(0), 0);
    }
    set.arrive_and_wait();
    int on = -1;
    EXPECT_EQ(tr_is_grad_enabled(&on), 0);
    seen[static_cast<size_t>(k)] = on;
  });
  EXPECT_EQ(seen[0], 0);
  EXPECT_EQ(seen[1], 1);
}

}  // namespace
