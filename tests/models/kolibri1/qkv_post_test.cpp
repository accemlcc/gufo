// Kolibri 1's per-head norm, rotation and cache write against an FP64 formula
// over the same inputs, so the only difference under test is the kernel's own
// arithmetic.
//
// What the cases pin: the per-head weights of Q and K rather than one norm over
// the projection, the pair (d, d + head_dim / 2) of the NEOX turn and the sign
// each half of the head uses, the layers that keep their queries and keys
// unturned, values passing through unchanged in width only, the ring slot of a
// position including the wrap inside one prefill, the grouped-query mapping at
// a shape where the last block is only partly full, and that a position at the
// end of the trained context still gets a turn a float angle could not hold.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include "src/models/kolibri1/kernels/rocm/qkv_post.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"

namespace k = gufo::models::kolibri1::rocm;

namespace {

int failures = 0;

constexpr double kTheta = 10000.0;
const double kThetaScale = std::pow(kTheta, -2.0 / 128.0);

std::uint16_t ToHalf(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, 2);
  return bits;
}

float FromHalf(std::uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return static_cast<float>(h);
}

template<class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

std::vector<std::uint16_t> Read(const std::uint16_t* device,
                                std::size_t count) {
  std::vector<std::uint16_t> host(count);
  HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  return host;
}

std::vector<float> Read(const float* device, std::size_t count) {
  std::vector<float> host(count);
  HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host;
}

struct Case {
  std::uint32_t rows;
  std::uint32_t first_position;
  std::uint32_t ring;  ///< cache capacity; 0 indexes by position
  std::uint32_t heads{48};
  std::uint32_t kv_heads{4};
  std::uint32_t head_dim{128};
  bool apply_rope{true};
  const char* name;
};

/// The oracle: norm one head, turn it, and say where it lands.
/// `out` holds head_dim values of a head, in double, exactly as the reference
/// implementation forms them.
void NormTurn(const std::vector<float>& x, const std::vector<float>& w,
              double eps, bool rope, std::uint32_t position,
              std::vector<double>* out) {
  const std::size_t dim = x.size();
  double ss = 0.0;
  for (const float v : x) {
    ss += static_cast<double>(v) * static_cast<double>(v);
  }
  const double r = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
  out->assign(dim, 0.0);
  for (std::size_t d = 0; d < dim; ++d) {
    (*out)[d] = static_cast<double>(x[d]) * r * static_cast<double>(w[d]);
  }
  if (!rope) {
    return;
  }
  const std::size_t half = dim / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const double angle = static_cast<double>(position) *
                         std::pow(kThetaScale, static_cast<double>(i));
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double a = (*out)[i];
    const double b = (*out)[i + half];
    (*out)[i] = a * c - b * s;
    (*out)[i + half] = a * s + b * c;
  }
}

struct Score {
  double max_abs = 0.0;
  double mean_rel = 0.0;
  double worst_at = 0.0;
};

Score Compare(const std::vector<double>& want, const std::vector<double>& got) {
  Score s;
  double total = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double diff = std::abs(got[i] - want[i]);
    total += diff / std::max(1e-3, std::abs(want[i]));
    if (diff > s.max_abs) {
      s.max_abs = diff;
      s.worst_at = want[i];
    }
  }
  s.mean_rel = total / static_cast<double>(want.size());
  return s;
}

void Check(const Case& c, const char* what, const std::vector<double>& want,
           const std::vector<double>& got, double max_abs, double mean_rel) {
  const Score s = Compare(want, got);
  if (s.max_abs > max_abs || s.mean_rel > mean_rel) {
    ++failures;
    std::cout << "FAIL " << c.name << " " << what << ": max_abs=" << s.max_abs
              << " mean_rel=" << s.mean_rel << " (at " << s.worst_at << ")\n";
    return;
  }
  std::cout << "PASS " << c.name << " " << what << ": max_abs=" << s.max_abs
            << " mean_rel=" << s.mean_rel << "\n";
}

void Run(const Case& c, std::mt19937& rng) {
  const std::uint32_t dim = c.head_dim;
  const std::size_t capacity =
      c.ring != 0 ? c.ring : std::size_t{c.first_position} + c.rows;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> q(std::size_t{c.rows} * c.heads * dim);
  std::vector<float> kk(std::size_t{c.rows} * c.kv_heads * dim);
  std::vector<float> v(kk.size());
  for (float& x : q) {
    x = normal(rng) * 2.0F;
  }
  for (float& x : kk) {
    x = normal(rng) * 2.0F;
  }
  for (float& x : v) {
    x = normal(rng) * 2.0F;
  }
  std::vector<float> q_norm(dim);
  std::vector<float> k_norm(dim);
  for (std::uint32_t d = 0; d < dim; ++d) {
    q_norm[d] = 0.5F + static_cast<float>(d) / static_cast<float>(dim);
    k_norm[d] = 1.5F - static_cast<float>(d) / static_cast<float>(2 * dim);
  }

  float* dq = Device(q);
  float* dk = Device(kk);
  float* dv = Device(v);
  float* dqn = Device(q_norm);
  float* dkn = Device(k_norm);
  std::vector<std::uint16_t> zeros(capacity * c.kv_heads * dim, 0);
  std::uint16_t* dkc = Device(zeros);
  std::uint16_t* dvc = Device(zeros);

  k::QkvPostArgs a{};
  a.q = dq;
  a.k = dk;
  a.v = dv;
  a.q_norm = dqn;
  a.k_norm = dkn;
  a.k_cache = dkc;
  a.v_cache = dvc;
  a.rows = c.rows;
  a.heads = c.heads;
  a.kv_heads = c.kv_heads;
  a.head_dim = dim;
  a.first_position = c.first_position;
  a.ring = c.ring;
  a.eps = 1e-6F;
  a.theta_scale = kThetaScale;
  a.apply_rope = c.apply_rope;
  k::QkvPost(a, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  const std::vector<float> got_q = Read(dq, q.size());
  const std::vector<std::uint16_t> got_kc = Read(dkc, zeros.size());
  const std::vector<std::uint16_t> got_vc = Read(dvc, zeros.size());

  // Expected values, written in the order the kernel writes them so a slot
  // that two positions share ends with the later one.
  std::vector<double> want_q(q.size());
  std::vector<double> want_kc(zeros.size(), 0.0);
  std::vector<double> want_vc(zeros.size(), 0.0);
  const auto slot = [&](std::uint32_t position) {
    return c.ring != 0 ? position % c.ring : position;
  };
  std::vector<double> head;
  for (std::uint32_t r = 0; r < c.rows; ++r) {
    const std::uint32_t position = c.first_position + r;
    for (std::uint32_t h = 0; h < c.heads; ++h) {
      const std::size_t base =
          (static_cast<std::size_t>(r) * c.heads + h) * dim;
      NormTurn({q.begin() + base, q.begin() + base + dim}, q_norm, 1e-6,
               c.apply_rope, position, &head);
      for (std::uint32_t d = 0; d < dim; ++d) {
        want_q[base + d] = head[d];
      }
    }
    for (std::uint32_t h = 0; h < c.kv_heads; ++h) {
      const std::size_t base =
          (static_cast<std::size_t>(r) * c.kv_heads + h) * dim;
      const std::size_t cache =
          (static_cast<std::size_t>(slot(position)) * c.kv_heads + h) * dim;
      NormTurn({kk.begin() + base, kk.begin() + base + dim}, k_norm, 1e-6,
               c.apply_rope, position, &head);
      for (std::uint32_t d = 0; d < dim; ++d) {
        want_kc[cache + d] = head[d];
      }
    }
    for (std::uint32_t h = 0; h < c.kv_heads; ++h) {
      const std::size_t base =
          (static_cast<std::size_t>(r) * c.kv_heads + h) * dim;
      const std::size_t cache =
          (static_cast<std::size_t>(slot(position)) * c.kv_heads + h) * dim;
      for (std::uint32_t d = 0; d < dim; ++d) {
        want_vc[cache + d] = FromHalf(ToHalf(v[base + d]));
      }
    }
  }

  // The cache holds binary16, so the rounding of a width change is part of the
  // expected result rather than an error under test.
  std::vector<double> got_qd(got_q.begin(), got_q.end());
  Check(c, "queries", want_q, got_qd, 2e-3, 1e-4);
  std::vector<double> got_kcd(got_kc.size());
  for (std::size_t i = 0; i < got_kc.size(); ++i) {
    got_kcd[i] = FromHalf(got_kc[i]);
  }
  Check(c, "keys", want_kc, got_kcd, 2e-2, 1e-3);
  std::vector<double> got_vcd(got_vc.size());
  for (std::size_t i = 0; i < got_vc.size(); ++i) {
    got_vcd[i] = FromHalf(got_vc[i]);
  }
  Check(c, "values", want_vc, got_vcd, 0.0, 0.0);

  for (void* ptr :
       {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
        static_cast<void*>(dqn), static_cast<void*>(dkn),
        static_cast<void*>(dkc), static_cast<void*>(dvc)}) {
    HIP_CHECK(hipFree(ptr));
  }
}

}  // namespace

int main() {
  std::mt19937 rng(20261012U);
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  Run({.rows = 1, .first_position = 40, .ring = 1024, .name = "decode"}, rng);
  Run({.rows = 1,
       .first_position = 4000,
       .ring = 8192,
       .apply_rope = false,
       .name = "layer without positional encoding"},
      rng);
  Run({.rows = 17, .first_position = 5, .ring = 4096, .name = "prefill"}, rng);
  Run({.rows = 8,
       .first_position = 12,
       .ring = 16,
       .name = "prefill across the ring wrap"},
      rng);
  // A partly full last block: ten heads over four waves per block means the
  // second block holds two active waves and two that must stay out of the way.
  Run({.rows = 3,
       .first_position = 1,
       .ring = 64,
       .heads = 6,
       .kv_heads = 2,
       .name = "grouped mapping at a small shape"},
      rng);
  // The end of the trained context, where a float angle has an ulp of 0.03 and
  // a cosine turns by that much.
  Run({.rows = 1,
       .first_position = 262143,
       .ring = 262144,
       .name = "position at the context limit"},
      rng);

  // Shapes the released model does not have are refused rather than
  // approximated.
  k::QkvPostArgs a{};
  a.rows = 1;
  a.heads = 48;
  a.kv_heads = 4;
  a.head_dim = 64;
  bool threw = false;
  try {
    k::QkvPost(a, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    ++failures;
    std::cout << "FAIL a head dimension other than 128 is accepted\n";
  } else {
    std::cout << "PASS a head dimension other than 128 is refused\n";
  }
  a.head_dim = 128;
  a.kv_heads = 5;
  threw = false;
  try {
    k::QkvPost(a, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    ++failures;
    std::cout
        << "FAIL query heads that are not a multiple of the key heads are "
           "accepted\n";
  } else {
    std::cout
        << "PASS query heads that are not a multiple of the key heads are "
           "refused\n";
  }

  if (failures != 0) {
    std::cout << failures << " qkv post checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 qkv post matches the reference over every case\n";
  return 0;
}
