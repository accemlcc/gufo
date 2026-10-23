// Qwen3.5-MoE's fused Q/K-norm/RoPE/split/cache-write against an FP64 CPU
// oracle over the same inputs. The oracle decodes nothing: every operand is
// float, so the only difference under test is the kernel's own arithmetic,
// and the packed [q0, g0, q1, g1, ...] deinterleave is pinned by crafted
// inputs whose gate halves read back exactly.
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
#include "src/models/qwen35moe/kernels/rocm/qkv_post.hpp"

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
constexpr std::uint32_t kRot = 64;
constexpr double kThetaScale = 1e-7;  // pow(1e7, -2/64)
constexpr float kEps = 1e-6F;

std::vector<float> CpuQkvPost(const std::vector<float>& qg,
                              const std::vector<float>& kk,
                              const std::vector<float>& vv,
                              const std::vector<float>& qn,
                              const std::vector<float>& kn,
                              std::uint32_t rows, std::uint32_t first_pos,
                              std::vector<float>& q_out,
                              std::vector<float>& gate_out,
                              std::vector<float>& k_cache,
                              std::vector<float>& v_cache,
                              std::uint32_t max_ctx) {
  const std::uint32_t qd = kHeads * kDim;
  q_out.assign(rows * qd, 0.0F);
  gate_out.assign(rows * qd, 0.0F);
  std::vector<double> fr(kRot / 2);
  for (std::uint32_t i = 0; i < kRot / 2; ++i) {
    fr[i] = std::pow(kThetaScale, static_cast<double>(i));
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint32_t pos = first_pos + r;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      double ss = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        const double v = qg[(r * 2 * qd) + h * 2 * kDim + i];
        ss += v * v;
      }
      const double s = 1.0 / std::sqrt(ss / kDim + kEps);
      for (std::uint32_t i = 0; i < kDim; ++i) {
        q_out[(r * kHeads + h) * kDim + i] =
            static_cast<float>(qg[(r * 2 * qd) + h * 2 * kDim + i] * s *
                               qn[i]);
        gate_out[(r * kHeads + h) * kDim + i] =
            qg[(r * 2 * qd) + h * 2 * kDim + kDim + i];
      }
      // Rotate the first 64 in pairs (i, i+32).
      for (std::uint32_t i = 0; i < kRot / 2; ++i) {
        const double ang = pos * fr[i];
        const double c = std::cos(ang), sn = std::sin(ang);
        const float a = q_out[(r * kHeads + h) * kDim + i];
        const float b = q_out[(r * kHeads + h) * kDim + i + kRot / 2];
        q_out[(r * kHeads + h) * kDim + i] = static_cast<float>(a * c - b * sn);
        q_out[(r * kHeads + h) * kDim + i + kRot / 2] =
            static_cast<float>(a * sn + b * c);
      }
    }
    for (std::uint32_t kh = 0; kh < kKv; ++kh) {
      double ss = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        const double v = kk[(r * kKv + kh) * kDim + i];
        ss += v * v;
      }
      const double s = 1.0 / std::sqrt(ss / kDim + kEps);
      for (std::uint32_t i = 0; i < kDim; ++i) {
        double v = kk[(r * kKv + kh) * kDim + i] * s * kn[i];
        k_cache[(kh * max_ctx + pos) * kDim + i] = static_cast<float>(v);
        v_cache[(kh * max_ctx + pos) * kDim + i] =
            vv[(r * kKv + kh) * kDim + i];
      }
      for (std::uint32_t i = 0; i < kRot / 2; ++i) {
        const double ang = pos * fr[i];
        const double c = std::cos(ang), sn = std::sin(ang);
        const float a = k_cache[(kh * max_ctx + pos) * kDim + i];
        const float b = k_cache[(kh * max_ctx + pos) * kDim + i + kRot / 2];
        k_cache[(kh * max_ctx + pos) * kDim + i] =
            static_cast<float>(a * c - b * sn);
        k_cache[(kh * max_ctx + pos) * kDim + i + kRot / 2] =
            static_cast<float>(a * sn + b * c);
      }
    }
  }
  return q_out;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

void RandomCase(std::uint32_t rows, std::uint32_t first_pos) {
  constexpr std::uint32_t kCtx = 64;
  const std::uint32_t qd = kHeads * kDim;
  const std::uint32_t kvd = kKv * kDim;
  std::mt19937 rng(100 + rows * 7 + first_pos);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> qg(rows * 2 * qd), kk(rows * kvd), vv(rows * kvd);
  std::vector<float> qn(kDim), kn(kDim);
  for (float& v : qg) {
    v = normal(rng);
  }
  for (float& v : kk) {
    v = normal(rng);
  }
  for (float& v : vv) {
    v = normal(rng);
  }
  for (float& v : qn) {
    v = 1.0F + 0.1F * normal(rng);
  }
  for (float& v : kn) {
    v = 1.0F + 0.1F * normal(rng);
  }
  std::vector<float> want_q, want_g, want_k(kKv * kCtx * kDim, 0.0F),
      want_v(kKv * kCtx * kDim, 0.0F);
  CpuQkvPost(qg, kk, vv, qn, kn, rows, first_pos, want_q, want_g, want_k,
             want_v, kCtx);
  std::vector<float> got_q(rows * qd), got_g(rows * qd),
      got_k(kKv * kCtx * kDim, 0.0F), got_v(kKv * kCtx * kDim, 0.0F);
  float* dqg = Device(qg);
  float* dkk = Device(kk);
  float* dvv = Device(vv);
  float* dqn = Device(qn);
  float* dkn = Device(kn);
  float* dq = Device(got_q);
  float* dg = Device(got_g);
  float* dk = Device(got_k);
  float* dv = Device(got_v);
  k::QkvPostArgs args{};
  args.qg = dqg;
  args.k = dkk;
  args.v = dvv;
  args.q_norm = dqn;
  args.k_norm = dkn;
  args.q_out = dq;
  args.gate_out = dg;
  args.k_cache = dk;
  args.v_cache = dv;
  args.rows = rows;
  args.first_position = first_pos;
  args.max_context = kCtx;
  args.heads = kHeads;
  args.kv_heads = kKv;
  args.dim = kDim;
  args.rotary_dim = kRot;
  args.theta_scale = kThetaScale;
  args.eps = kEps;
  k::QkvPost(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  got_q = Read(dq, got_q);
  got_g = Read(dg, got_g);
  got_k = Read(dk, got_k);
  got_v = Read(dv, got_v);
  const double dqv = MaxAbsDiff(got_q, want_q);
  const double dgv = MaxAbsDiff(got_g, want_g);
  const double dkv = MaxAbsDiff(got_k, want_k);
  const double dvv = MaxAbsDiff(got_v, want_v);
  const double worst = std::max(std::max(dqv, dgv), std::max(dkv, dvv));
  const bool ok = worst < 2e-5;
  Report(("qkv_post.random_r" + std::to_string(rows) + "_p" +
          std::to_string(first_pos))
             .c_str(),
         ok,
         "worst=" + std::to_string(worst));
  HIP_CHECK(hipFree(dqg));
  HIP_CHECK(hipFree(dkk));
  HIP_CHECK(hipFree(dvv));
  HIP_CHECK(hipFree(dqn));
  HIP_CHECK(hipFree(dkn));
  HIP_CHECK(hipFree(dq));
  HIP_CHECK(hipFree(dg));
  HIP_CHECK(hipFree(dk));
  HIP_CHECK(hipFree(dv));
}

void InterleavePin() {
  // Crafted packed input: head h reads h * 1000 + i, gate half +0.5.
  // A halves split would put whole heads in the wrong half; the deinterleave
  // puts every value where it belongs, including the gate copy.
  constexpr std::uint32_t kCtx = 8;
  const std::uint32_t qd = kHeads * kDim;
  const std::uint32_t rows = 1;
  std::vector<float> qg(2 * qd), kk(kKv * kDim, 0.0F), vv(kKv * kDim, 0.0F),
      qn(kDim, 1.0F), kn(kDim, 1.0F);
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    for (std::uint32_t i = 0; i < kDim; ++i) {
      qg[h * 2 * kDim + i] = static_cast<float>(h * 1000 + i);
      qg[h * 2 * kDim + kDim + i] = static_cast<float>(h * 1000 + i) + 0.5F;
    }
  }
  std::vector<float> want_q, want_g, want_k(kKv * kCtx * kDim, 0.0F),
      want_v(kKv * kCtx * kDim, 0.0F);
  CpuQkvPost(qg, kk, vv, qn, kn, rows, 0, want_q, want_g, want_k, want_v, kCtx);
  std::vector<float> got_q(qd), got_g(qd),
      got_k(kKv * kCtx * kDim, 0.0F), got_v(kKv * kCtx * kDim, 0.0F);
  float* dqg = Device(qg);
  float* dkk = Device(kk);
  float* dvv = Device(vv);
  float* dqn = Device(qn);
  float* dkn = Device(kn);
  float* dq = Device(got_q);
  float* dg = Device(got_g);
  float* dk = Device(got_k);
  float* dv = Device(got_v);
  k::QkvPostArgs args{};
  args.qg = dqg;
  args.k = dkk;
  args.v = dvv;
  args.q_norm = dqn;
  args.k_norm = dkn;
  args.q_out = dq;
  args.gate_out = dg;
  args.k_cache = dk;
  args.v_cache = dv;
  args.rows = rows;
  args.first_position = 0;
  args.max_context = kCtx;
  args.heads = kHeads;
  args.kv_heads = kKv;
  args.dim = kDim;
  args.rotary_dim = kRot;
  args.theta_scale = kThetaScale;
  args.eps = kEps;
  k::QkvPost(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  got_q = Read(dq, got_q);
  got_g = Read(dg, got_g);
  const double worst =
      std::max(MaxAbsDiff(got_q, want_q), MaxAbsDiff(got_g, want_g));
  // Spot check: gate of head 5 starts at 5000.5 (pos 0: rope is identity,
  // and the gate is copied without a norm). The query half is normalized, so
  // only its oracle comparison counts.
  const bool spot = std::fabs(got_g[5 * kDim] - 5000.5F) < 1e-3;
  Report("qkv_post.interleave", worst < 2e-5 && spot,
         "worst=" + std::to_string(worst));
  HIP_CHECK(hipFree(dqg));
  HIP_CHECK(hipFree(dkk));
  HIP_CHECK(hipFree(dvv));
  HIP_CHECK(hipFree(dqn));
  HIP_CHECK(hipFree(dkn));
  HIP_CHECK(hipFree(dq));
  HIP_CHECK(hipFree(dg));
  HIP_CHECK(hipFree(dk));
  HIP_CHECK(hipFree(dv));
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
  RandomCase(4, 0);
  RandomCase(4, 17);
  InterleavePin();

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
