// Qwen3.5-MoE's online-softmax GQA attention against an FP64 CPU oracle over
// the same cache: decode rows and prefill chunks, adjacent head groups, with
// crafted key/value heads that tell the groups apart.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen35moe/kernels/rocm/attention.hpp"

namespace k = gufo::models::qwen35moe::rocm;

namespace {

int failures = 0;

void Report(const char* name, bool ok, const std::string& detail) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": " << detail << "\n";
  if (!ok) {
    ++failures;
  }
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

constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kKv = 2;
constexpr std::uint32_t kDim = 256;
constexpr float kScale = 1.0F / 16.0F;
constexpr std::uint32_t kCtx = 64;

std::vector<float> CpuAttention(const std::vector<float>& q,
                                const std::vector<float>& kc,
                                const std::vector<float>& vc,
                                std::uint32_t rows, std::uint32_t first_pos) {
  const std::uint32_t group = kHeads / kKv;
  std::vector<float> out(rows * kHeads * kDim, 0.0F);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint32_t hi = first_pos + r + 1;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / group;
      double best = -1e300, total = 0.0;
      std::vector<double> sc(hi);
      for (std::uint32_t p = 0; p < hi; ++p) {
        double s = 0.0;
        for (std::uint32_t i = 0; i < kDim; ++i) {
          s += static_cast<double>(
                   q[(r * kHeads + h) * kDim + i]) *
               kc[(kvh * kCtx + p) * kDim + i];
        }
        s *= kScale;
        sc[p] = s;
        if (s > best) {
          best = s;
        }
      }
      for (std::uint32_t p = 0; p < hi; ++p) {
        total += std::exp(sc[p] - best);
      }
      for (std::uint32_t i = 0; i < kDim; ++i) {
        double acc = 0.0;
        for (std::uint32_t p = 0; p < hi; ++p) {
          acc += std::exp(sc[p] - best) / total *
                 vc[(kvh * kCtx + p) * kDim + i];
        }
        out[(r * kHeads + h) * kDim + i] = static_cast<float>(acc);
      }
    }
  }
  return out;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

void RandomCase(std::uint32_t rows, std::uint32_t first_pos) {
  std::mt19937 rng(200 + rows * 13 + first_pos);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> q(rows * kHeads * kDim);
  std::vector<float> kc(kKv * kCtx * kDim), vc(kKv * kCtx * kDim);
  for (float& v : q) {
    v = normal(rng);
  }
  for (float& v : kc) {
    v = normal(rng);
  }
  for (float& v : vc) {
    v = normal(rng);
  }
  std::vector<float> want = CpuAttention(q, kc, vc, rows, first_pos);
  std::vector<float> got(rows * kHeads * kDim);
  float* dq = Device(q);
  float* dkc = Device(kc);
  float* dvc = Device(vc);
  float* dout = Device(got);
  k::AttentionArgs args{};
  args.q = dq;
  args.k_cache = dkc;
  args.v_cache = dvc;
  args.out = dout;
  args.rows = rows;
  args.first_position = first_pos;
  args.max_context = kCtx;
  args.heads = kHeads;
  args.kv_heads = kKv;
  args.dim = kDim;
  args.scale = kScale;
  k::Attention(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  got = Read(dout, got);
  const double worst = MaxAbsDiff(got, want);
  Report(("attention.random_r" + std::to_string(rows) + "_p" +
          std::to_string(first_pos))
             .c_str(),
         worst < 2e-5, "worst=" + std::to_string(worst));
  HIP_CHECK(hipFree(dq));
  HIP_CHECK(hipFree(dkc));
  HIP_CHECK(hipFree(dvc));
  HIP_CHECK(hipFree(dout));
}

void GroupPin() {
  // kv head 0 is all +1, kv head 1 all -1; queries all +1. Heads 0..7 must
  // read +1 values, heads 8..15 the -1 values: the adjacent grouping made
  // visible in one row.
  constexpr std::uint32_t kRows = 1;
  std::vector<float> q(kRows * kHeads * kDim, 1.0F);
  std::vector<float> kc(kKv * kCtx * kDim), vc(kKv * kCtx * kDim);
  for (std::uint32_t i = 0; i < kKv * kCtx * kDim; ++i) {
    kc[i] = 0.0F;
  }
  for (std::uint32_t p = 0; p < kCtx; ++p) {
    for (std::uint32_t i = 0; i < kDim; ++i) {
      vc[(0 * kCtx + p) * kDim + i] = 1.0F;
      vc[(1 * kCtx + p) * kDim + i] = -1.0F;
    }
  }
  // Only position 0 is attended (first_position 0, one row).
  std::vector<float> want = CpuAttention(q, kc, vc, kRows, 0);
  std::vector<float> got(kRows * kHeads * kDim);
  float* dq = Device(q);
  float* dkc = Device(kc);
  float* dvc = Device(vc);
  float* dout = Device(got);
  k::AttentionArgs args{};
  args.q = dq;
  args.k_cache = dkc;
  args.v_cache = dvc;
  args.out = dout;
  args.rows = kRows;
  args.first_position = 0;
  args.max_context = kCtx;
  args.heads = kHeads;
  args.kv_heads = kKv;
  args.dim = kDim;
  args.scale = kScale;
  k::Attention(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  got = Read(dout, got);
  bool ok = MaxAbsDiff(got, want) < 1e-6;
  // Heads 0..7 see +1, heads 8..15 see -1.
  ok = ok && got[0] > 0.99F && got[8 * kDim] < -0.99F;
  Report("attention.groups", ok, "adjacent kv pairing");
  HIP_CHECK(hipFree(dq));
  HIP_CHECK(hipFree(dkc));
  HIP_CHECK(hipFree(dvc));
  HIP_CHECK(hipFree(dout));
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  RandomCase(1, 0);
  RandomCase(1, 37);
  RandomCase(4, 0);
  RandomCase(4, 10);
  GroupPin();

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
