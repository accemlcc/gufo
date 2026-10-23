// Qwen3.5-MoE's block arithmetic against an FP64 formula: the only
// difference under test is the kernel's own rounding. RmsNorm reads the same
// row once whatever the batch, MoeFinish folds the shared row in first and
// adds experts in increasing order, so a row keeps its value when a batch
// boundary moves it.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"

namespace k = gufo::models::qwen35moe::rocm;

namespace {

int failures = 0;

void Report(const char* name, bool ok, const std::string& detail) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": " << detail << "\n";
  if (!ok) {
    ++failures;
  }
}

std::string Number(double v) {
  std::ostringstream s;
  s << std::setprecision(4) << v;
  return s.str();
}

template<class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

template<class T>
std::vector<T> Read(T* device, std::vector<T> host) {
  HIP_CHECK(hipMemcpy(host.data(), device, host.size() * sizeof(T),
                      hipMemcpyDeviceToHost));
  return host;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

void NormRows() {
  constexpr std::uint32_t kRows = 5;
  constexpr std::uint32_t kN = 2048;
  constexpr float kEps = 1e-6F;
  std::mt19937 rng(21);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(kRows * kN), w(kN);
  for (float& v : x) {
    v = normal(rng);
  }
  for (float& v : w) {
    v = 1.0F + 0.1F * normal(rng);
  }
  std::vector<float> want(kRows * kN);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double ss = 0.0;
    for (std::uint32_t i = 0; i < kN; ++i) {
      const double v = x[r * kN + i];
      ss += v * v;
    }
    const double s = 1.0 / std::sqrt(ss / kN + kEps);
    for (std::uint32_t i = 0; i < kN; ++i) {
      want[r * kN + i] = static_cast<float>(x[r * kN + i] * s * w[i]);
    }
  }
  float* dx = Device(x);
  float* dw = Device(w);
  std::vector<float> y(kRows * kN);
  float* dy = Device(y);
  k::NormRowsArgs args{};
  args.x = dx;
  args.w = dw;
  args.y = dy;
  args.rows = kRows;
  args.n = kN;
  args.eps = kEps;
  k::RmsNormRows(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  y = Read(dy, y);
  const double d = MaxAbsDiff(y, want);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dy));
  Report("blocks.norm", d < 2e-6, "max_abs=" + Number(d));
}

void AddGateUp() {
  constexpr std::size_t kCount = 2048 + 3;  // with a scalar tail
  std::mt19937 rng(22);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(kCount), add(kCount), gate(kCount), up(kCount);
  for (float& v : x) {
    v = normal(rng);
  }
  for (float& v : add) {
    v = normal(rng);
  }
  for (float& v : gate) {
    v = normal(rng);
  }
  for (float& v : up) {
    v = normal(rng);
  }
  std::vector<float> want_add = x, want_act(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    want_add[i] += add[i];
    const double g = gate[i];
    want_act[i] =
        static_cast<float>(g / (1.0 + std::exp(-g)) * up[i]);
  }
  float* dx = Device(x);
  float* dadd = Device(add);
  k::AddRowsArgs aargs{};
  aargs.x = dx;
  aargs.add = dadd;
  aargs.count = kCount;
  k::AddRows(aargs, nullptr);
  float* dgate = Device(gate);
  float* dup = Device(up);
  std::vector<float> act(kCount);
  float* dact = Device(act);
  k::GateUpArgs gargs{};
  gargs.gate = dgate;
  gargs.up = dup;
  gargs.act = dact;
  gargs.count = kCount;
  k::GateUpAct(gargs, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  x = Read(dx, x);
  act = Read(dact, act);
  const double d0 = MaxAbsDiff(x, want_add);
  const double d1 = MaxAbsDiff(act, want_act);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dadd));
  HIP_CHECK(hipFree(dgate));
  HIP_CHECK(hipFree(dup));
  HIP_CHECK(hipFree(dact));
  Report("blocks.add_gateup", d0 == 0.0 && d1 < 2e-6,
         "add=" + Number(d0) + " act=" + Number(d1));
}

void Finish() {
  constexpr std::uint32_t kRows = 3;
  constexpr std::uint32_t kHidden = 2048;
  constexpr std::uint32_t kUsed = 8;
  constexpr float kEps = 1e-6F;
  std::mt19937 rng(23);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(kRows * kHidden);
  std::vector<float> routed(kRows * kUsed * kHidden);
  std::vector<float> shared(kRows * kHidden);
  std::vector<float> weights(kRows * kUsed);
  std::vector<float> post(kHidden);
  for (float& v : x) {
    v = normal(rng);
  }
  for (float& v : routed) {
    v = normal(rng);
  }
  for (float& v : shared) {
    v = normal(rng);
  }
  for (float& v : weights) {
    v = normal(rng);
  }
  for (float& v : post) {
    v = 1.0F + 0.1F * normal(rng);
  }
  // Renormalise the weights the way the router leaves them.
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double s = 0.0;
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      s += std::exp(weights[r * kUsed + j]);
    }
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      weights[r * kUsed + j] =
          static_cast<float>(std::exp(weights[r * kUsed + j]) / s);
    }
  }
  std::vector<float> want = x;
  for (std::uint32_t r = 0; r < kRows; ++r) {
    std::vector<double> mix(kHidden);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      mix[i] = shared[r * kHidden + i];
      for (std::uint32_t j = 0; j < kUsed; ++j) {
        mix[i] += weights[r * kUsed + j] *
                  routed[(r * kUsed + j) * kHidden + i];
      }
    }
    double ss = 0.0;
    for (double v : mix) {
      ss += v * v;
    }
    const double s = 1.0 / std::sqrt(ss / kHidden + kEps);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      want[r * kHidden + i] += static_cast<float>(mix[i] * s * post[i]);
    }
  }
  float* dx = Device(x);
  float* drouted = Device(routed);
  float* dshared = Device(shared);
  float* dweights = Device(weights);
  float* dpost = Device(post);
  k::MoeFinishArgs args{};
  args.x = dx;
  args.routed = drouted;
  args.shared = dshared;
  args.weights = dweights;
  args.post_norm = dpost;
  args.rows = kRows;
  args.hidden = kHidden;
  args.used = kUsed;
  args.eps = kEps;
  k::MoeFinish(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  x = Read(dx, x);
  const double d = MaxAbsDiff(x, want);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(drouted));
  HIP_CHECK(hipFree(dshared));
  HIP_CHECK(hipFree(dweights));
  HIP_CHECK(hipFree(dpost));
  // The kernel reads `x` back and the want holds the same float rounding the
  // reference takes, so this is tight; the engine contract only needs 0.5.
  Report("blocks.finish", d < 1e-4, "max_abs=" + Number(d));
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  NormRows();
  AddGateUp();
  Finish();

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
