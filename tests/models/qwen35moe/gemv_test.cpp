// Qwen3.5-MoE's grouped projection against an FP64 dot of the same bytes the
// kernel reads: the weights are built as raw quantized blocks whose codes are
// pseudo-random and whose floating-point scale fields are small positives, and
// the expectation decodes them with the CPU decoder the engine loads with. The
// only difference under test is therefore the kernel's own decode and the
// order it adds in. The Q5_K lane is new here; Q4_K/Q6_K follow the same
// structure.
//
// What the cases pin: every format the artifacts carry, a dense single-group
// projection with one row and with sixteen, a table where one expert holds
// slots of several rows and another holds none, and that one slot comes out
// bit-identical whether it runs alone or in a table with others.
//
// Needs a gfx1151 device; without one the check skips with 77.
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
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"

namespace k = gufo::models::qwen35moe::rocm;
namespace q = gufo::quant;

namespace {

int failures = 0;

constexpr float kPoison = -1.0e37F;

void Report(const char* name, bool ok, const std::string& detail) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": " << detail << "\n";
  if (!ok) {
    ++failures;
  }
}

std::uint16_t ToHalf(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, 2);
  return bits;
}

std::vector<void*> g_live;

template<class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  g_live.push_back(ptr);
  return ptr;
}

template<class T>
std::vector<T> Read(T* device, std::vector<T> host) {
  HIP_CHECK(hipMemcpy(host.data(), device, host.size() * sizeof(T),
                      hipMemcpyDeviceToHost));
  return host;
}

void FreeAll() {
  for (void* p : g_live) {
    HIP_CHECK(hipFree(p));
  }
  g_live.clear();
}

std::size_t BlockBytes(k::WeightFormat format) {
  switch (format) {
    case k::WeightFormat::kQ4_K:
      return sizeof(q::block_q4_K);
    case k::WeightFormat::kQ5_K:
      return sizeof(q::block_q5_K);
    case k::WeightFormat::kQ6_K:
      return sizeof(q::block_q6_K);
    case k::WeightFormat::kQ8_0:
      return sizeof(q::block_q8_0);
  }
  return 0;
}

std::size_t RowBytes(k::WeightFormat format, std::uint32_t kdim) {
  return static_cast<std::size_t>(kdim) / 256 * BlockBytes(format);
}

struct Built {
  std::vector<std::uint8_t> bytes;
  std::vector<float> decoded;
};

Built Build(k::WeightFormat format, std::uint32_t experts, std::uint32_t m,
            std::uint32_t kdim, std::mt19937& rng) {
  const std::size_t row = RowBytes(format, kdim);
  const std::size_t bb = BlockBytes(format);
  Built w;
  w.bytes.assign(static_cast<std::size_t>(experts) * m * row, 0);
  std::uniform_int_distribution<int> raw(0, 255);
  for (auto& v : w.bytes) {
    v = static_cast<std::uint8_t>(raw(rng));
  }
  std::uniform_real_distribution<float> scale(0.001F, 0.006F);
  for (std::size_t r = 0; r < static_cast<std::size_t>(experts) * m; ++r) {
    for (std::uint32_t b = 0; b < kdim / 256; ++b) {
      std::uint8_t* p =
          w.bytes.data() + r * row + static_cast<std::size_t>(b) * bb;
      if (format == k::WeightFormat::kQ4_K) {
        auto* blk = reinterpret_cast<q::block_q4_K*>(p);
        blk->d = ToHalf(scale(rng));
        blk->dmin = ToHalf(0.5F * scale(rng));
      } else if (format == k::WeightFormat::kQ5_K) {
        auto* blk = reinterpret_cast<q::block_q5_K*>(p);
        blk->d = ToHalf(scale(rng));
        blk->dmin = ToHalf(0.5F * scale(rng));
      } else if (format == k::WeightFormat::kQ8_0) {
        reinterpret_cast<q::block_q8_0*>(p)->d = ToHalf(scale(rng));
      } else {
        reinterpret_cast<q::block_q6_K*>(p)->d = ToHalf(0.25F * scale(rng));
      }
    }
  }
  w.decoded.resize(static_cast<std::size_t>(experts) * m * kdim);
  for (std::size_t r = 0; r < static_cast<std::size_t>(experts) * m; ++r) {
    float* dst = w.decoded.data() + r * kdim;
    const void* src = w.bytes.data() + r * row;
    if (format == k::WeightFormat::kQ4_K) {
      q::DequantizeQ4_K(src, dst, kdim);
    } else if (format == k::WeightFormat::kQ5_K) {
      q::DequantizeQ5_K(src, dst, kdim);
    } else if (format == k::WeightFormat::kQ8_0) {
      q::DequantizeQ8_0(src, dst, kdim);
    } else {
      q::DequantizeQ6_K(src, dst, kdim);
    }
  }
  return w;
}

/// FP64 oracle dot of one decoded row.
double Dot(const float* w, const float* x, std::uint32_t kdim) {
  double acc = 0.0;
  for (std::uint32_t i = 0; i < kdim; ++i) {
    acc += static_cast<double>(w[i]) * x[i];
  }
  return acc;
}

struct Case {
  k::WeightFormat format;
  std::uint32_t experts;
  std::uint32_t m;
  std::uint32_t kdim;
  std::uint32_t rows;
  std::uint32_t slots_per_row;
  // Groups as (expert, slots...) with slot = row * slots_per_row + s.
  std::vector<std::vector<std::int32_t>> table;
};

std::string CaseName(const Case& c) {
  const char* f = c.format == k::WeightFormat::kQ4_K      ? "q4k"
                    : c.format == k::WeightFormat::kQ5_K ? "q5k"
                    : c.format == k::WeightFormat::kQ8_0 ? "q8_0"
                                                         : "q6k";
  return std::string(f) + "_e" + std::to_string(c.experts) + "_m" +
         std::to_string(c.m) + "_k" + std::to_string(c.kdim) + "_r" +
         std::to_string(c.rows);
}

void RunCase(const Case& c, std::mt19937& rng, bool& ok, std::string& detail) {
  Built w = Build(c.format, c.experts, c.m, c.kdim, rng);
  std::vector<float> x(c.rows * c.kdim);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (float& v : x) {
    v = normal(rng);
  }
  // Flatten the group table.
  std::uint32_t ngroups = static_cast<std::uint32_t>(c.table.size());
  std::vector<std::int32_t> groups(k::GroupTableInts(ngroups), -1);
  groups[0] = static_cast<std::int32_t>(ngroups);
  std::uint32_t nslots = 0;
  for (std::uint32_t gi = 0; gi < ngroups; ++gi) {
    std::int32_t* g = groups.data() + 1 + gi * k::kGroupInts;
    g[0] = c.table[gi][0];
    const std::uint32_t n =
        static_cast<std::uint32_t>(c.table[gi].size()) - 1;
    g[1] = static_cast<std::int32_t>(n);
    for (std::uint32_t s = 0; s < n; ++s) {
      g[2 + s] = c.table[gi][1 + s];
    }
    nslots = std::max(nslots, n);
    (void)nslots;
  }
  std::uint32_t total_slots = 0;
  for (const auto& g : c.table) {
    total_slots += static_cast<std::uint32_t>(g.size()) - 1;
  }
  std::vector<float> y(total_slots * c.m, kPoison);
  // Oracle per slot in FP64.
  std::vector<double> want(total_slots * c.m);
  {
    std::uint32_t slot = 0;
    for (const auto& g : c.table) {
      const std::int32_t e = g[0];
      for (std::size_t s = 1; s < g.size(); ++s, ++slot) {
        const std::uint32_t r =
            static_cast<std::uint32_t>(g[s]) / c.slots_per_row;
        for (std::uint32_t i = 0; i < c.m; ++i) {
          const float* wrow = w.decoded.data() +
                              (static_cast<std::size_t>(e) * c.m + i) *
                                  c.kdim;
          want[static_cast<std::size_t>(slot) * c.m + i] =
              Dot(wrow, x.data() + static_cast<std::size_t>(r) * c.kdim,
                  c.kdim);
        }
      }
    }
  }
  uint8_t* dw = Device(w.bytes);
  float* dx = Device(x);
  float* dy = Device(y);
  std::int32_t* dgroups = Device(groups);
  k::GroupedGemvArgs args{};
  args.format = c.format;
  args.w = dw;
  args.groups = dgroups;
  args.max_groups = ngroups;
  args.x = dx;
  args.y = dy;
  args.rows = c.rows;
  args.slots_per_row = c.slots_per_row;
  args.m = c.m;
  args.k = c.kdim;
  ok = k::GroupedGemv(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  y = Read(dy, y);
  FreeAll();
  if (!ok) {
    detail = "launch refused";
    return;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(y[i]) - want[i]));
  }
  // FP64 oracle against the kernel's float FMA chains: the bound is the
  // kolibri one, 16 * eps * bound, with the row's own magnitude.
  double bound = 0.0;
  for (double v : want) {
    bound = std::max(bound, std::fabs(v));
  }
  ok = worst <= 16 * 1.19e-7 * std::max(bound, 1.0) * c.kdim / 64;
  detail = "worst=" + std::to_string(worst);
}

void DenseAndGrouped() {
  std::mt19937 rng(31);
  const k::WeightFormat formats[4] = {
      k::WeightFormat::kQ4_K, k::WeightFormat::kQ5_K,
      k::WeightFormat::kQ6_K, k::WeightFormat::kQ8_0};
  for (k::WeightFormat f : formats) {
    // Dense single-group, one row and sixteen.
    for (std::uint32_t rows : {1U, 16U}) {
      Case c{f, 1, 128, 512, rows, 1, {}};
      c.table.push_back({0});
      for (std::uint32_t r = 0; r < rows; ++r) {
        c.table[0].push_back(static_cast<std::int32_t>(r));
      }
      bool ok = false;
      std::string detail;
      RunCase(c, rng, ok, detail);
      Report(("gemv." + CaseName(c)).c_str(), ok, detail);
    }
    // Two experts, slots shared across rows; every slot index stays below
    // rows * slots_per_row.
    {
      Case c{f, 3, 64, 512, 2, 2, {{1, 0, 1, 2}, {0, 3}}};
      bool ok = false;
      std::string detail;
      RunCase(c, rng, ok, detail);
      Report(("gemv.grouped_" + CaseName(c)).c_str(), ok, detail);
    }
  }
}

void BitInvariance() {
  // One slot alone and inside a wider table: identical bits.
  std::mt19937 rng(32);
  Built w = Build(k::WeightFormat::kQ5_K, 2, 64, 512, rng);
  std::vector<float> x(2 * 512);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (float& v : x) {
    v = normal(rng);
  }
  std::vector<std::int32_t> solo_groups(k::GroupTableInts(1), -1);
  solo_groups[0] = 1;
  solo_groups[1] = 1;
  solo_groups[2] = 2;
  solo_groups[3] = 0;
  std::vector<std::int32_t> wide_groups(k::GroupTableInts(2), -1);
  wide_groups[0] = 2;
  wide_groups[1] = 1;
  wide_groups[2] = 2;
  wide_groups[3] = 0;
  wide_groups[4] = 1;
  wide_groups[1 + k::kGroupInts] = 0;
  wide_groups[1 + k::kGroupInts + 1] = 3;
  wide_groups[1 + k::kGroupInts + 2] = 1;
  wide_groups[1 + k::kGroupInts + 3] = 2;
  wide_groups[1 + k::kGroupInts + 4] = 3;
  std::vector<float> solo(1 * 64, kPoison), wide(4 * 64, kPoison);
  uint8_t* dw = Device(w.bytes);
  float* dx = Device(x);
  float* dsolo = Device(solo);
  float* dwide = Device(wide);
  std::int32_t* dsg = Device(solo_groups);
  std::int32_t* dwg = Device(wide_groups);
  k::GroupedGemvArgs args{};
  args.format = k::WeightFormat::kQ5_K;
  args.w = dw;
  args.x = dx;
  args.rows = 2;
  args.slots_per_row = 2;
  args.m = 64;
  args.k = 512;
  args.groups = dsg;
  args.max_groups = 1;
  args.y = dsolo;
  bool ok_solo = k::GroupedGemv(args, nullptr);
  args.groups = dwg;
  args.max_groups = 2;
  args.y = dwide;
  bool ok_wide = k::GroupedGemv(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  solo = Read(dsolo, solo);
  wide = Read(dwide, wide);
  FreeAll();
  bool ok = ok_solo && ok_wide;
  // Slot 0 of the wide table is the same (expert 1, row 0, slot 0).
  for (std::uint32_t i = 0; ok && i < 64; ++i) {
    std::uint32_t a = 0, b = 0;
    std::memcpy(&a, &solo[i], 4);
    std::memcpy(&b, &wide[i], 4);
    ok = a == b;
  }
  Report("gemv.bit_invariance", ok, "q5k slot identical solo and grouped");
}

void Refusals() {
  bool ok = true;
  k::GroupedGemvArgs args{};
  args.format = k::WeightFormat::kQ5_K;
  args.rows = 1;
  args.slots_per_row = 1;
  args.m = 64;
  args.k = 512;
  args.max_groups = 1;
  if (k::GroupedGemv(args, nullptr)) {
    ok = false;  // null pointers must refuse
  }
  args.k = 100;
  if (k::GroupedGemv(args, nullptr)) {
    ok = false;  // k outside the blocks must refuse
  }
  Report("gemv.refusals", ok, "null and ragged widths refused");
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  DenseAndGrouped();
  BitInvariance();
  Refusals();

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
