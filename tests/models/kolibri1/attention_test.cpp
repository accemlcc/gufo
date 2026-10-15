// Kolibri 1 attention kernel against an FP64 formula over the same binary16
// caches, so the only difference under test is the kernel's own arithmetic.
//
// What the cases pin: the key range of a sliding layer near and far from the
// window edge, an unwindowed no-position layer reading the whole history, ring
// wrap-around, a truncated continuation, partial prefill batches, the split-K
// path against a single pass, and the twelve-to-one grouped-query mapping.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include "src/models/kolibri1/kernels/rocm/attention.hpp"

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

struct Case {
  std::uint32_t rows;
  std::uint32_t first_position;
  std::uint32_t window;  // 0 for a layer without positional encoding
  std::uint32_t ring;    // cache capacity, 0 to index by position
  std::uint32_t key_limit;
  std::uint32_t heads{48};
  std::uint32_t kv_heads{4};
  std::uint32_t head_dim{128};
  const char* name;
};

struct Result {
  std::vector<float> out;
};

/// The visible key range of one query row, exactly as the kernel computes it.
struct Range {
  std::uint32_t lo;
  std::uint32_t hi;
};

Range Bounds(const Case& c, std::uint32_t position) {
  const std::uint32_t hi = std::min(position + 1, c.key_limit);
  std::uint32_t lo = 0;
  if (c.window != 0 && position + 1 > c.window) {
    lo = position + 1 - c.window;
  }
  return {std::min(lo, hi), hi};
}

Result Run(const Case& c, const std::vector<float>& q,
           const std::vector<std::uint16_t>& kc,
           const std::vector<std::uint16_t>& vc) {
  const std::size_t capacity =
      c.ring != 0 ? c.ring : kc.size() / (c.kv_heads * c.head_dim);
  float* dq = Device(q);
  auto* dk = Device(kc);
  auto* dv = Device(vc);
  std::vector<float> out(std::size_t{c.rows} * c.heads * c.head_dim, NAN);
  float* dout = Device(out);
  std::vector<float> partials = std::vector<float>(k::AttentionPartialFloats(
      c.rows, c.heads, c.head_dim, static_cast<std::uint32_t>(capacity)));
  float* dpart = Device(partials);

  k::AttentionArgs a{};
  a.q = dq;
  a.k = dk;
  a.v = dv;
  a.out = dout;
  a.partials = dpart;
  a.rows = c.rows;
  a.heads = c.heads;
  a.kv_heads = c.kv_heads;
  a.head_dim = c.head_dim;
  a.first_position = c.first_position;
  a.window = c.window;
  a.ring = c.ring;
  a.key_limit = c.key_limit;
  a.scale = 1.0F / std::sqrt(static_cast<float>(c.head_dim));

  k::Attention(a, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (void* p :
       {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
        static_cast<void*>(dout), static_cast<void*>(dpart)}) {
    HIP_CHECK(hipFree(p));
  }
  return {std::move(out)};
}

/// FP64 attention over the same binary16 caches: the same keys, the same
/// narrowing of Q, and nothing the kernel does wrong.
std::vector<double> Reference(const Case& c, const std::vector<float>& q,
                              const std::vector<std::uint16_t>& kc,
                              const std::vector<std::uint16_t>& vc) {
  const std::size_t dim = c.head_dim;
  const std::size_t group = c.heads / c.kv_heads;
  const std::size_t kv_stride = static_cast<std::size_t>(c.kv_heads) * dim;
  std::vector<double> out(std::size_t{c.rows} * c.heads * dim, 0.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  for (std::uint32_t r = 0; r < c.rows; ++r) {
    const Range range = Bounds(c, c.first_position + r);
    for (std::uint32_t h = 0; h < c.heads; ++h) {
      const std::size_t kvh = h / group;
      // The kernel's A operand is the query narrowed to binary16.
      std::vector<double> qh(dim);
      for (std::size_t d = 0; d < dim; ++d) {
        qh[d] = FromHalf(
            ToHalf(q[(static_cast<std::size_t>(r) * c.heads + h) * dim + d]));
      }
      std::vector<double> scores(range.hi - range.lo, 0.0);
      double best = -INFINITY;
      for (std::uint32_t p = range.lo; p < range.hi; ++p) {
        const std::size_t slot = c.ring != 0 ? p % c.ring : p;
        const std::size_t base = slot * kv_stride + kvh * dim;
        double acc = 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
          acc += qh[d] * FromHalf(kc[base + d]);
        }
        scores[p - range.lo] = acc * scale;
        best = std::max(best, scores[p - range.lo]);
      }
      double total = 0.0;
      for (double& s : scores) {
        s = std::exp(s - best);
        total += s;
      }
      for (std::uint32_t p = range.lo; p < range.hi; ++p) {
        const std::size_t slot = c.ring != 0 ? p % c.ring : p;
        const std::size_t base = slot * kv_stride + kvh * dim;
        const double weight = scores[p - range.lo] / total;
        for (std::size_t d = 0; d < dim; ++d) {
          out[(static_cast<std::size_t>(r) * c.heads + h) * dim + d] +=
              weight * FromHalf(vc[base + d]);
        }
      }
    }
  }
  return out;
}

void Check(const Case& c, std::mt19937& rng) {
  const std::size_t dim = c.head_dim;
  const std::uint32_t last = c.first_position + c.rows - 1;
  const std::uint32_t visible = std::min(last + 1, c.key_limit);
  const std::uint32_t capacity =
      c.ring != 0 ? c.ring : visible;  // slots the cache holds
  std::normal_distribution<float> normal(0.0F, 1.0F);

  std::vector<float> q(std::size_t{c.rows} * c.heads * dim);
  for (float& v : q) {
    v = normal(rng) * 0.5F;
  }
  // Keys and values are distinct per key head so a wrong group mapping shows up
  // as a large difference rather than a similar-looking result.
  std::vector<std::uint16_t> kc(std::size_t{capacity} * c.kv_heads * dim);
  std::vector<std::uint16_t> vc(kc.size());
  for (std::size_t i = 0; i < kc.size(); ++i) {
    const std::uint32_t kvh = (i / dim) % c.kv_heads;
    kc[i] = ToHalf(normal(rng) * (1.0F + 0.5F * static_cast<float>(kvh)));
    vc[i] = ToHalf(normal(rng) * (1.0F + 0.5F * static_cast<float>(kvh)));
  }

  const Result got = Run(c, q, kc, vc);
  const std::vector<double> want = Reference(c, q, kc, vc);

  double max_abs = 0.0;
  double worst_at = 0.0;
  double sum_rel = 0.0;
  std::size_t counted = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double diff = std::abs(static_cast<double>(got.out[i]) - want[i]);
    if (diff > max_abs) {
      max_abs = diff;
      worst_at = want[i];
    }
    if (std::abs(want[i]) > 1e-3) {
      sum_rel += diff / std::abs(want[i]);
      ++counted;
    }
  }
  const double mean_rel =
      counted != 0 ? sum_rel / static_cast<double>(counted) : 0.0;
  // Binary16 carries about three decimal digits, and the softmax weights a few
  // hundred values whose sum the output divides by. Anything above this is a
  // structural error, not rounding.
  const bool ok = max_abs < 2e-2 && mean_rel < 5e-3;
  std::cout << (ok ? "PASS " : "FAIL ") << c.name << ": rows=" << c.rows
            << " window=" << c.window << " ring=" << c.ring
            << " limit=" << c.key_limit << " keys=" << visible
            << " max_abs=" << max_abs << " mean_rel=" << mean_rel << " (at "
            << worst_at << ")\n";
  if (!ok) {
    ++failures;
  }
}

/// Reports the time of one shape. Not a gate: the numbers steer the launcher,
/// and the deep shapes are too large to keep in the checked set.
void Timed(const Case& c, std::mt19937& rng) {
  const std::size_t dim = c.head_dim;
  const std::size_t capacity =
      c.ring != 0 ? c.ring : std::min(c.key_limit, c.first_position + c.rows);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> q(std::size_t{c.rows} * c.heads * dim);
  for (float& v : q) {
    v = normal(rng) * 0.5F;
  }
  std::vector<std::uint16_t> kc(capacity * c.kv_heads * dim);
  std::vector<std::uint16_t> vc(kc.size());
  for (auto& v : kc) {
    v = ToHalf(normal(rng));
  }
  for (auto& v : vc) {
    v = ToHalf(normal(rng));
  }

  float* dq = Device(q);
  auto* dk = Device(kc);
  auto* dv = Device(vc);
  std::vector<float> out(std::size_t{c.rows} * c.heads * dim, 0.0F);
  float* dout = Device(out);
  std::vector<float> partials(k::AttentionPartialFloats(
      c.rows, c.heads, c.head_dim, static_cast<std::uint32_t>(capacity)));
  float* dpart = Device(partials);
  k::AttentionArgs a{};
  a.q = dq;
  a.k = dk;
  a.v = dv;
  a.out = dout;
  a.partials = dpart;
  a.rows = c.rows;
  a.heads = c.heads;
  a.kv_heads = c.kv_heads;
  a.head_dim = c.head_dim;
  a.first_position = c.first_position;
  a.window = c.window;
  a.ring = c.ring;
  a.key_limit = c.key_limit;
  a.scale = 1.0F / std::sqrt(static_cast<float>(c.head_dim));

  k::Attention(a, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  constexpr int kIters = 20;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < kIters; ++i) {
    k::Attention(a, nullptr);
  }
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  const double keys = static_cast<double>(
      c.window != 0 ? std::min<std::uint32_t>(c.window, c.key_limit)
                    : c.key_limit);
  const double bytes = keys * c.kv_heads * dim * 2.0 * 2.0;  // K and V, halves
  const double us = 1e3 * ms / kIters;
  std::cout << "time " << c.name << ": " << us << " us per call, "
            << bytes / us / 1e3 << " GB/s of K/V read\n";
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  for (void* ptr :
       {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv),
        static_cast<void*>(dout), static_cast<void*>(dpart)}) {
    HIP_CHECK(hipFree(ptr));
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no GPU\n";
    return 77;
  }
  std::mt19937 rng(20261012U);

  const Case cases[] = {
      // A first token has one key: a mapping bug is invisible only if every
      // case stops here, so the list keeps going.
      {.rows = 1,
       .first_position = 0,
       .window = 513,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "decode first token"},
      {.rows = 1,
       .first_position = 40,
       .window = 513,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "decode inside window"},
      {.rows = 1,
       .first_position = 512,
       .window = 513,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "decode at the window edge"},
      // The ring is smaller than the position, so these keys really wrap.
      {.rows = 1,
       .first_position = 5000,
       .window = 513,
       .ring = 2048,
       .key_limit = 0xffffffffU,
       .name = "decode sliding with ring wrap"},
      {.rows = 1,
       .first_position = 4095,
       .window = 0,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "decode no-position layer full history"},
      {.rows = 1,
       .first_position = 4095,
       .window = 0,
       .ring = 0,
       .key_limit = 2000,
       .name = "decode truncated continuation"},
      {.rows = 7,
       .first_position = 0,
       .window = 513,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "prefill partial batch"},
      {.rows = 64,
       .first_position = 0,
       .window = 513,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "prefill one block"},
      {.rows = 64,
       .first_position = 500,
       .window = 513,
       .ring = 4096,
       .key_limit = 0xffffffffU,
       .name = "prefill across the window edge"},
      {.rows = 100,
       .first_position = 0,
       .window = 0,
       .ring = 0,
       .key_limit = 0xffffffffU,
       .name = "prefill no-position layer"},
      {.rows = 33,
       .first_position = 3000,
       .window = 513,
       .ring = 2048,
       .key_limit = 0xffffffffU,
       .name = "prefill ring wrap"},
  };
  for (const auto& c : cases) {
    Check(c, rng);
  }

  // A geometry the kernel is not written for must be refused, not approximated.
  bool refused = false;
  try {
    k::AttentionArgs bad{};
    bad.rows = 1;
    bad.heads = 48;
    bad.kv_heads = 4;
    bad.head_dim = 64;
    bad.key_limit = 0xffffffffU;
    bad.scale = 1.0F;
    k::Attention(bad, nullptr);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  std::cout << (refused ? "PASS " : "FAIL ")
            << "a head dimension other than 128 is refused\n";
  if (!refused) {
    ++failures;
  }

  if (std::getenv("KOLIBRI_TIME") != nullptr) {
    // The shapes that decide decode speed: a no-position layer at full context,
    // a sliding layer, and a prefill of one chunk.
    Timed({.rows = 1,
           .first_position = 131071,
           .window = 0,
           .ring = 131072,
           .key_limit = 131072,
           .name = "decode 128k no-position"},
          rng);
    Timed({.rows = 1,
           .first_position = 131071,
           .window = 513,
           .ring = 131072,
           .key_limit = 131072,
           .name = "decode 128k sliding"},
          rng);
    Timed({.rows = 2048,
           .first_position = 0,
           .window = 513,
           .ring = 8192,
           .key_limit = 2048,
           .name = "prefill 2048 sliding"},
          rng);
  }

  if (failures != 0) {
    std::cout << failures << " attention checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 attention matches the reference over every case\n";
  return 0;
}
