// Kolibri 1's grouped projection against an FP64 dot of the same bytes the
// kernel reads: the weights are built as raw quantized blocks whose codes are
// pseudo-random and whose floating-point scale fields are small positives, and
// the expectation decodes them with the CPU decoder the engine loads with. The
// only difference under test is therefore the kernel's own decode and the order
// it adds in.
//
// What the cases pin: both formats the artifact carries, a dense single-group
// projection at every width the layer stack uses with one row and with sixteen,
// a table where one expert holds slots of several rows and another holds none,
// a group order that is not the slot order, a group that holds nothing, an
// output row no group claims, and that one slot comes out bit-identical whether
// it runs alone, in a pass of four, in a pass of eight, or in a table with
// fifteen others.
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
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"

namespace k = gufo::models::kolibri1::rocm;
namespace q = gufo::quant;

namespace {

int failures = 0;

/// A value no dot of these weights can produce, so an output the table does not
/// claim is recognizable as untouched.
constexpr float kPoison = -1.0e37F;

std::uint16_t ToHalf(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, 2);
  return bits;
}

bool SameBits(float a, float b) {
  std::uint32_t x = 0, y = 0;
  std::memcpy(&x, &a, 4);
  std::memcpy(&y, &b, 4);
  return x == y;
}

// The buffers of one case are as large as a projection tensor, and three agents
// share this GPU, so they go back the way they came.
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

/// A buffer of raw bytes, for the refusal checks, which need a pointer the
/// allocator's alignment never hands out.
unsigned char* RawDevice(std::size_t bytes) {
  unsigned char* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, bytes + 16));
  g_live.push_back(ptr);
  return ptr;
}

void FreeAll() {
  for (void* p : g_live) {
    HIP_CHECK(hipFree(p));
  }
  g_live.clear();
}

/// Row bytes of a format, taken from the layout header rather than repeated
/// here: the sizes are the ones the CPU decoder walks with.
std::size_t RowBytes(k::WeightFormat format, std::uint32_t k) {
  const std::size_t block = format == k::WeightFormat::kQ4_K
                                ? sizeof(q::block_q4_K)
                                : sizeof(q::block_q6_K);
  return static_cast<std::size_t>(k) / 256 * block;
}

struct Built {
  std::vector<std::uint8_t> bytes;  ///< [experts][m][row bytes]
  std::vector<float> decoded;       ///< [experts][m][k], by the CPU decoder
};

/// Random codes everywhere, then the scale fields of every block replaced by
/// small positives: a random half would be an infinity, and the products have
/// to stay near one for the reported errors to mean anything. `decode` fills
/// the FP32 copy the oracle dots against; a shape that is only timed skips it.
Built Build(k::WeightFormat format, std::uint32_t experts, std::uint32_t m,
            std::uint32_t kdim, std::mt19937& rng, bool decode = true) {
  const std::size_t row = RowBytes(format, kdim);
  const std::size_t block_bytes = format == k::WeightFormat::kQ4_K
                                      ? sizeof(q::block_q4_K)
                                      : sizeof(q::block_q6_K);
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
          w.bytes.data() + r * row + static_cast<std::size_t>(b) * block_bytes;
      if (format == k::WeightFormat::kQ4_K) {
        auto* blk = reinterpret_cast<q::block_q4_K*>(p);
        blk->d = ToHalf(scale(rng));
        blk->dmin = ToHalf(0.5F * scale(rng));
      } else {
        // Q6_K's scales field is integer and its one float is the block's d.
        reinterpret_cast<q::block_q6_K*>(p)->d = ToHalf(0.25F * scale(rng));
      }
    }
  }
  if (!decode) {
    return w;
  }
  w.decoded.resize(static_cast<std::size_t>(experts) * m * kdim);
  for (std::size_t r = 0; r < static_cast<std::size_t>(experts) * m; ++r) {
    float* dst = w.decoded.data() + r * kdim;
    const std::uint8_t* src = w.bytes.data() + r * row;
    if (format == k::WeightFormat::kQ4_K) {
      q::DequantizeQ4_K(src, dst, kdim);
    } else {
      q::DequantizeQ6_K(src, dst, kdim);
    }
  }
  return w;
}

struct Group {
  std::int32_t expert;
  std::vector<std::int32_t> slots;
};

std::vector<std::int32_t> MakeTable(const std::vector<Group>& groups) {
  std::vector<std::int32_t> table(
      k::GroupTableInts(static_cast<std::uint32_t>(groups.size())), 0);
  table[0] = static_cast<std::int32_t>(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::int32_t* p = table.data() + 1 + g * k::kGroupInts;
    p[0] = groups[g].expert;
    p[1] = static_cast<std::int32_t>(groups[g].slots.size());
    for (std::size_t i = 0; i < groups[g].slots.size(); ++i) {
      p[2 + i] = groups[g].slots[i];
    }
  }
  return table;
}

/// One launch and its output, with nothing checked: the invariant cases run the
/// same weights and activations through several tables and compare the runs.
std::vector<float> Launch(const Built& w, const std::vector<float>& x,
                          const std::vector<Group>& groups,
                          k::WeightFormat format, std::uint32_t rows,
                          std::uint32_t slots_per_row, std::uint32_t m,
                          std::uint32_t kdim, std::uint32_t max_groups) {
  const std::vector<std::int32_t> table = MakeTable(groups);
  std::vector<float> out(static_cast<std::size_t>(rows) * slots_per_row * m,
                         kPoison);
  const std::uint8_t* dw = Device(w.bytes);
  const float* dx = Device(x);
  const std::int32_t* dg = Device(table);
  float* dy = Device(out);
  k::GroupedGemvArgs args{};
  args.format = format;
  args.w = dw;
  args.groups = dg;
  args.max_groups = max_groups;
  args.x = dx;
  args.y = dy;
  args.rows = rows;
  args.slots_per_row = slots_per_row;
  args.m = m;
  args.k = kdim;
  if (!k::GroupedGemv(args, nullptr)) {
    std::cout << "  launcher refused the shape\n";
    ++failures;
    return out;
  }
  HIP_CHECK(hipStreamSynchronize(nullptr));
  HIP_CHECK(hipMemcpy(out.data(), dy, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  return out;
}

struct GemvCase {
  const char* name;
  k::WeightFormat format;
  std::uint32_t experts;
  std::uint32_t m;
  std::uint32_t kdim;
  std::uint32_t rows;
  std::uint32_t slots_per_row;
  std::uint32_t max_groups;
  std::vector<Group> groups;
};

/// The FP64 dot of one slot against one row of the activation, plus the sum of
/// the absolute products, which is the scale an error of this addition order
/// has to be measured against: a dot whose terms cancel has no meaningful
/// relative error of its own.
void Oracle(const float* w, const float* xv, std::uint32_t kdim, double* want,
            double* bound) {
  double sum = 0.0;
  double abs_sum = 0.0;
  for (std::uint32_t i = 0; i < kdim; ++i) {
    const double p = static_cast<double>(w[i]) * static_cast<double>(xv[i]);
    sum += p;
    abs_sum += std::fabs(p);
  }
  *want = sum;
  *bound = abs_sum;
}

void Check(const GemvCase& c, std::mt19937& rng) {
  const Built w = Build(c.format, c.experts, c.m, c.kdim, rng);
  std::vector<float> x(static_cast<std::size_t>(c.rows) * c.kdim);
  std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
  for (auto& v : x) {
    v = unit(rng);
  }
  const std::vector<float> out =
      Launch(w, x, c.groups, c.format, c.rows, c.slots_per_row, c.m, c.kdim,
             c.max_groups);

  const std::uint32_t slots = c.rows * c.slots_per_row;
  std::vector<char> claimed(slots, 0);
  // Sixteen ulps of the absolute bound: the decode difference between this
  // file's fused products and the CPU decoder's unfused ones is one ulp of a
  // weight, and the addition order is worth a few more. A wrong block, scale,
  // stride or expert is orders of magnitude outside it.
  const double tol = 16.0 * std::numeric_limits<float>::epsilon();
  double max_abs = 0.0;
  double sum_rel = 0.0;
  std::size_t checked = 0, wrong = 0, leaked = 0;
  for (const Group& g : c.groups) {
    for (const std::int32_t s : g.slots) {
      claimed[s] = 1;
      const float* wr =
          w.decoded.data() + static_cast<std::size_t>(g.expert) * c.m * c.kdim;
      const float* xv =
          x.data() + static_cast<std::size_t>(s / c.slots_per_row) * c.kdim;
      for (std::uint32_t row = 0; row < c.m; ++row) {
        double want = 0.0, bound = 0.0;
        Oracle(wr + static_cast<std::size_t>(row) * c.kdim, xv, c.kdim, &want,
               &bound);
        const float got = out[static_cast<std::size_t>(s) * c.m + row];
        const double err = std::fabs(static_cast<double>(got) - want);
        max_abs = std::max(max_abs, err);
        sum_rel += err / std::max(1.0e-9, std::fabs(want));
        ++checked;
        if (!(err <= tol * bound)) {
          ++wrong;
        }
      }
    }
  }
  // A slot no group holds must keep what the caller left there.
  for (std::uint32_t s = 0; s < slots; ++s) {
    if (claimed[s]) {
      continue;
    }
    for (std::uint32_t row = 0; row < c.m; ++row) {
      if (!SameBits(out[static_cast<std::size_t>(s) * c.m + row], kPoison)) {
        ++leaked;
      }
    }
  }
  const bool ok = wrong == 0 && leaked == 0 && checked != 0;
  std::cout << (ok ? "PASS " : "FAIL ") << c.name << "  values " << checked
            << "  max abs " << max_abs << "  mean rel "
            << (checked ? sum_rel / static_cast<double>(checked) : 0.0)
            << (leaked ? "  wrote an unclaimed row" : "") << "\n";
  if (!ok) {
    ++failures;
  }
  FreeAll();
}

/// One slot of one launch, alone in the table or among fifteen others, at every
/// pass width and split the launcher can pick for the shape: the same bits, or
/// a batch boundary moves a token.
void CheckInvariance(k::WeightFormat format, std::uint32_t m,
                     std::uint32_t kdim, std::mt19937& rng) {
  constexpr std::uint32_t kRows = 16;
  const Built w = Build(format, 2, m, kdim, rng);
  std::vector<float> x(static_cast<std::size_t>(kRows) * kdim);
  std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
  for (auto& v : x) {
    v = unit(rng);
  }
  const char* fmt = format == k::WeightFormat::kQ4_K ? "Q4_K" : "Q6_K";
  char name[160];

  // Sixteen rows of dense activation, one expert: all in one group, in four
  // groups, and each slot alone in a launch of its own.
  std::vector<Group> all{{0, {}}};
  for (std::int32_t s = 0; s < static_cast<std::int32_t>(kRows); ++s) {
    all[0].slots.push_back(s);
  }
  std::vector<Group> split;
  for (std::int32_t base = 0; base < static_cast<std::int32_t>(kRows);
       base += 4) {
    Group g{0, {}};
    for (std::int32_t s = base; s < base + 4; ++s) {
      g.slots.push_back(s);
    }
    split.push_back(g);
  }
  const std::vector<float> wide =
      Launch(w, x, all, format, kRows, 1, m, kdim, 1);
  const std::vector<float> parts =
      Launch(w, x, split, format, kRows, 1, m, kdim, 4);

  std::size_t differ = 0;
  for (std::uint32_t s = 0; s < kRows; ++s) {
    // The row count is what the launcher picks its pass width from, so a solo
    // launch of s + 1 rows runs the same chain at a width the wide launch
    // never uses for that slot.
    const std::vector<float> solo =
        Launch(w, x, {{0, {static_cast<std::int32_t>(s)}}}, format, s + 1, 1, m,
               kdim, 1);
    // A launch indexes its output by slot, so the run of slot s alone leaves
    // everything before it as it was: the window to compare is s * m in both.
    for (std::uint32_t row = 0; row < m; ++row) {
      const std::size_t i = static_cast<std::size_t>(s) * m + row;
      if (!SameBits(wide[i], parts[i]) || !SameBits(wide[i], solo[i])) {
        ++differ;
      }
    }
  }
  std::snprintf(name, sizeof(name),
                "%s slot alone against sixteen rows and four groups m %u k %u",
                fmt, m, kdim);
  std::cout << (differ == 0 ? "PASS " : "FAIL ") << name << "  differing "
            << differ << "\n";
  if (differ != 0) {
    ++failures;
  }

  // One slot against every pass width the launcher can choose, so the widths
  // are compared with each other and not only with their own launch.
  std::size_t moved = 0;
  for (const std::uint32_t rows : {1U, 2U, 5U, 16U, 33U, 128U}) {
    const std::vector<float> one =
        Launch(w, x, {{0, {0}}}, format, rows, 1, m, kdim, 1);
    for (std::uint32_t row = 0; row < m; ++row) {
      if (!SameBits(wide[row], one[row])) {
        ++moved;
      }
    }
  }
  std::snprintf(name, sizeof(name),
                "%s one slot at pass widths one, four, eight and sixteen m %u "
                "k %u",
                fmt, m, kdim);
  std::cout << (moved == 0 ? "PASS " : "FAIL ") << name << "  differing "
            << moved << "\n";
  if (moved != 0) {
    ++failures;
  }

  // The routed shape: sixteen slots of three rows in two groups that each span
  // all three rows, against a single slot of one of those groups held alone at
  // the pass width a wider batch would pick.
  constexpr std::uint32_t kSpr = 6;
  std::vector<Group> routed{{0, {}}, {1, {}}};
  for (std::int32_t s = 0; s < static_cast<std::int32_t>(kRows); ++s) {
    routed[s % 2].slots.push_back(s);
  }
  const std::vector<float> mixed =
      Launch(w, x, routed, format, 3, kSpr, m, kdim, 2);
  std::size_t shifted = 0;
  for (const std::int32_t s : {1, 8, 15}) {
    for (const std::uint32_t rows : {3U, 16U}) {
      const std::vector<float> solo =
          Launch(w, x, {{s % 2, {s}}}, format, rows, kSpr, m, kdim, 1);
      for (std::uint32_t row = 0; row < m; ++row) {
        const std::size_t i = static_cast<std::size_t>(s) * m + row;
        if (!SameBits(mixed[i], solo[i])) {
          ++shifted;
        }
      }
    }
  }
  std::snprintf(name, sizeof(name),
                "%s slot of a three-row six-expert table held alone m %u k %u",
                fmt, m, kdim);
  std::cout << (shifted == 0 ? "PASS " : "FAIL ") << name << "  differing "
            << shifted << "\n";
  if (shifted != 0) {
    ++failures;
  }

  // A group wider than the pass a launch makes: forty-eight rows' slots on two
  // experts, which is the shape the routed projections of a whole forward
  // present and one a table built inside one selection block could never list.
  // Three launches of the same slots have to agree bit for bit: the wide group
  // at the pass width of a hundred and twenty-eight rows, the same group at the
  // pass width of eight, and the slots in a group each of their own - which is
  // the shape the single-slot case already pins against the FP64 dot, so
  // agreeing with it carries that pin over to the wide group.
  constexpr std::uint32_t kDeep = 48;
  std::vector<float> xdeep(static_cast<std::size_t>(k::kMaxGroupSlots) * kdim);
  for (auto& v : xdeep) {
    v = unit(rng);
  }
  std::vector<Group> deep{{0, {}}, {1, {}}};
  std::vector<Group> singletons;
  for (std::int32_t s = 0; s < static_cast<std::int32_t>(kDeep); ++s) {
    deep[s % 2].slots.push_back(s);
    singletons.push_back({s % 2, {s}});
  }
  const std::vector<float> wide_group =
      Launch(w, xdeep, deep, format, k::kMaxGroupSlots, kSpr, m, kdim, 2);
  const std::vector<float> narrow_group =
      Launch(w, xdeep, deep, format, 8, kSpr, m, kdim, 2);
  const std::vector<float> apart = Launch(
      w, xdeep, singletons, format, k::kMaxGroupSlots, kSpr, m, kdim, kDeep);
  std::size_t crossed = 0;
  for (std::int32_t s = 0; s < static_cast<std::int32_t>(kDeep); ++s) {
    for (std::uint32_t row = 0; row < m; ++row) {
      const std::size_t i = static_cast<std::size_t>(s) * m + row;
      if (!SameBits(wide_group[i], narrow_group[i]) ||
          !SameBits(wide_group[i], apart[i])) {
        ++crossed;
      }
    }
  }
  std::snprintf(name, sizeof(name),
                "%s forty-eight slots in one group across pass widths m %u "
                "k %u",
                fmt, m, kdim);
  std::cout << (crossed == 0 ? "PASS " : "FAIL ") << name << "  differing "
            << crossed << "\n";
  if (crossed != 0) {
    ++failures;
  }
}

void CheckEmbed(k::WeightFormat format, std::uint32_t vocab, std::uint32_t n,
                std::mt19937& rng) {
  const Built w = Build(format, 1, vocab, n, rng);
  std::vector<std::int32_t> tokens;
  std::uniform_int_distribution<std::int32_t> pick(
      0, static_cast<std::int32_t>(vocab) - 1);
  for (int i = 0; i < 9; ++i) {
    tokens.push_back(pick(rng));
  }
  tokens[0] = 0;
  tokens[1] = static_cast<std::int32_t>(vocab) - 1;
  tokens[2] = tokens[1];  // the same row twice, from two positions

  std::vector<float> out(tokens.size() * n, kPoison);
  const std::uint8_t* dt = Device(w.bytes);
  const std::int32_t* dtok = Device(tokens);
  float* dy = Device(out);
  k::EmbedArgs args{};
  args.table = dt;
  args.format = format;
  args.tokens = dtok;
  args.y = dy;
  args.rows = static_cast<std::uint32_t>(tokens.size());
  args.n = n;
  bool threw = false;
  try {
    k::EmbedRows(args, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  HIP_CHECK(hipStreamSynchronize(nullptr));
  HIP_CHECK(hipMemcpy(out.data(), dy, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  double max_abs = 0.0, sum_rel = 0.0;
  std::size_t wrong = 0;
  for (std::size_t r = 0; r < tokens.size(); ++r) {
    const float* want =
        w.decoded.data() + static_cast<std::size_t>(tokens[r]) * n;
    // The width of the terms the decode multiplied: a value that happens to
    // cancel is only as right as the products that made it.
    double scale = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
      scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
    }
    for (std::uint32_t i = 0; i < n; ++i) {
      const double ref = static_cast<double>(want[i]);
      const double err = std::fabs(static_cast<double>(out[r * n + i]) - ref);
      max_abs = std::max(max_abs, err);
      sum_rel += err / std::max(1.0e-9, std::fabs(ref));
      // Nothing is added here, only the fused products of the decode, so the
      // gap is a rounding of the value and not of a sum.
      if (!(err <= 8.0 * std::numeric_limits<float>::epsilon() *
                           (std::fabs(ref) + scale) +
                       1.0e-12)) {
        ++wrong;
      }
    }
  }
  const bool ok = !threw && wrong == 0;
  std::cout << (ok ? "PASS " : "FAIL ")
            << (format == k::WeightFormat::kQ4_K ? "Q4_K" : "Q6_K")
            << " embedding gather vocab " << vocab << " n " << n << "  values "
            << tokens.size() * n << "  max abs " << max_abs << "  mean rel "
            << sum_rel / static_cast<double>(tokens.size() * n) << "\n";
  if (!ok) {
    ++failures;
  }
  FreeAll();
}

std::vector<std::int32_t> Count(std::int32_t from, std::int32_t to) {
  std::vector<std::int32_t> v;
  for (std::int32_t i = from; i < to; ++i) {
    v.push_back(i);
  }
  return v;
}

/// Reports the time of one launch and the weight bytes it moves. Not a gate:
/// the numbers steer the launcher, and here they answer one question for the
/// engine, whether binding a launch to every expert of the model instead of to
/// the groups the table holds costs a decode step. A shared GPU reads these
/// high rather than exact.
///
/// Two lines where a rotation is possible, because one table is not a decode
/// step: the rounds read the same experts again and again, and the 4.42 MB a
/// six-slot gate launch moves fits this generation's L2, so the first number
/// is a cache number — 318 GB/s there against 187 for the same launch over
/// tables that cannot be cached (BENCHMARKS: decode's routed stage). The second
/// line runs each round on experts no earlier round read, until the round
/// streams about `kRotatingBytes`; a shape moving that much in one launch keeps
/// its single line.
void Timed(const char* name, k::WeightFormat format, std::uint32_t experts,
           std::uint32_t m, std::uint32_t kdim, std::uint32_t rows,
           std::uint32_t slots_per_row, const std::vector<Group>& groups,
           std::uint32_t max_groups, std::mt19937& rng) {
  const Built w = Build(format, experts, m, kdim, rng, false);
  std::vector<float> x(static_cast<std::size_t>(rows) * kdim, 0.25F);
  const std::vector<std::int32_t> table = MakeTable(groups);
  std::vector<float> out(static_cast<std::size_t>(rows) * slots_per_row * m,
                         0.0F);
  k::GroupedGemvArgs args{};
  args.format = format;
  args.w = Device(w.bytes);
  args.groups = Device(table);
  args.max_groups = max_groups;
  args.x = Device(x);
  args.y = Device(out);
  args.rows = rows;
  args.slots_per_row = slots_per_row;
  args.m = m;
  args.k = kdim;
  const auto launch_all = [&](int rounds) -> bool {
    for (int i = 0; i < rounds; ++i) {
      if (!k::GroupedGemv(args, nullptr)) {
        return false;
      }
    }
    return true;
  };
  if (!launch_all(2)) {
    std::cout << "TIME " << name << "  refused\n";
    ++failures;
    FreeAll();
    return;
  }
  constexpr int kRounds = 20;
  const auto Clock = [&](const auto& run) -> float {
    run(2);
    HIP_CHECK(hipStreamSynchronize(nullptr));
    hipEvent_t first = nullptr, last = nullptr;
    HIP_CHECK(hipEventCreate(&first));
    HIP_CHECK(hipEventCreate(&last));
    HIP_CHECK(hipEventRecord(first, nullptr));
    const bool ran = run(kRounds);
    HIP_CHECK(hipEventRecord(last, nullptr));
    HIP_CHECK(hipEventSynchronize(last));
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, first, last));
    HIP_CHECK(hipEventDestroy(first));
    HIP_CHECK(hipEventDestroy(last));
    if (!ran) {
      ++failures;
    }
    return ms;
  };
  const double read = static_cast<double>(groups.size()) *
                      static_cast<double>(m) *
                      static_cast<double>(RowBytes(format, kdim));
  const auto report = [&](const char* tail, float ms) {
    const double us = ms * 1000.0 / kRounds;
    std::cout << "TIME " << name << "  " << us << " us  "
              << read / (us * 1.0e-6) / 1.0e9 << " GB/s of claimed experts, "
              << tail << "\n";
  };
  report("one table", Clock(launch_all));

  // A rotation replaces each group's expert instead of shifting it: the decode
  // case's six ids span the model on purpose (0, 37, 91, 140, 255, 383), so a
  // shift would leave the model after one round. Expert `r * touched + g` is
  // read by no other round, and a round's own ids stay in increasing order. A
  // table already holding one group per expert has nothing to rotate through
  // and keeps its single line. What the rotated round gains against the model
  // is that its experts are neighbours in the file, where a decode row's six
  // are scattered; its number is therefore a ceiling on the cold rate.
  const std::uint32_t touched = static_cast<std::uint32_t>(groups.size());
  const std::uint32_t room =
      touched > 0U && touched <= experts ? experts / touched : 1U;
  constexpr std::size_t kRotatingBytes = 256ULL << 20;
  const std::size_t per_round =
      std::max<std::size_t>(1, static_cast<std::size_t>(read));
  const std::uint32_t rotations = std::max(
      1U,
      std::min(room, static_cast<std::uint32_t>(kRotatingBytes / per_round)));
  if (rotations <= 1U) {
    FreeAll();
    return;
  }
  const std::size_t stride = table.size();
  std::vector<std::int32_t> tables(stride * rotations, 0);
  for (std::uint32_t r = 0; r < rotations; ++r) {
    std::int32_t* t = tables.data() + r * stride;
    std::memcpy(t, table.data(), stride * sizeof(std::int32_t));
    for (std::uint32_t g = 0; g < touched; ++g) {
      t[1 + g * k::kGroupInts] = static_cast<std::int32_t>(r * touched + g);
    }
  }
  const std::int32_t* rotating = Device(tables);
  // The table is all a rotation changes: the groups keep their slots, so every
  // round multiplies the same activations by weights nothing has read since
  // that expert was last touched, which is the access a decode step makes.
  const auto launch_rotating = [&](int rounds) -> bool {
    for (int i = 0; i < rounds; ++i) {
      args.groups = rotating + static_cast<std::size_t>(i % rotations) * stride;
      if (!k::GroupedGemv(args, nullptr)) {
        return false;
      }
    }
    return true;
  };
  const std::string tail = std::to_string(rotations) + " disjoint tables";
  report(tail.c_str(), Clock(launch_rotating));
  FreeAll();
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }
  std::mt19937 rng(20261012U);

  const std::pair<std::uint32_t, std::uint32_t> shapes[] = {
      {6144, 2560}, {512, 2560}, {2560, 6144}, {1024, 2560}, {2560, 512},
  };
  const k::WeightFormat formats[] = {k::WeightFormat::kQ4_K,
                                     k::WeightFormat::kQ6_K};

  // A dense projection is the grouped one with one expert and the rows as its
  // slots, so it goes through the same kernel and the same table.
  for (const k::WeightFormat f : formats) {
    for (const auto [m, kdim] : shapes) {
      for (const std::uint32_t rows : {1U, 16U}) {
        const char* fmt = f == k::WeightFormat::kQ4_K ? "Q4_K" : "Q6_K";
        char name[128];
        std::snprintf(name, sizeof(name), "dense %s m %u k %u rows %u", fmt, m,
                      kdim, rows);
        Check({.name = name,
               .format = f,
               .experts = 1,
               .m = m,
               .kdim = kdim,
               .rows = rows,
               .slots_per_row = 1,
               .max_groups = 1,
               .groups = {{0, Count(0, static_cast<std::int32_t>(rows))}}},
              rng);
      }
    }
  }

  // Six expert tensors; the table claims three of them, lists their slots in an
  // order that is not the slot order, hands one group nothing, and leaves one
  // slot of the batch unclaimed. max_groups is larger than the count, so blocks
  // without a group have to stay out of the way.
  for (const k::WeightFormat f : formats) {
    const char* fmt = f == k::WeightFormat::kQ4_K ? "Q4_K" : "Q6_K";
    const std::vector<Group> groups{
        {5, {9, 17}},
        {2, {0, 1, 6, 12, 15}},
        {4, {2, 3, 4, 7, 8, 10, 11, 13, 14, 16}},
        {3, {}},
    };
    char name[128];
    std::snprintf(name, sizeof(name),
                  "grouped %s three experts of three rows, one empty", fmt);
    Check({.name = name,
           .format = f,
           .experts = 6,
           .m = 512,
           .kdim = 2560,
           .rows = 3,
           .slots_per_row = 6,
           .max_groups = 24,
           .groups = groups},
          rng);
    std::snprintf(name, sizeof(name),
                  "grouped %s narrow down projection m 2560 k 512", fmt);
    Check({.name = name,
           .format = f,
           .experts = 6,
           .m = 2560,
           .kdim = 512,
           .rows = 3,
           .slots_per_row = 6,
           .max_groups = 24,
           .groups = groups},
          rng);
  }

  // The batch boundary must not move a value: 512x2560 runs on the split
  // reduction and 2560x512 on whole rows per wave.
  for (const k::WeightFormat f : formats) {
    CheckInvariance(f, 512, 2560, rng);
    CheckInvariance(f, 2560, 512, rng);
  }

  CheckEmbed(k::WeightFormat::kQ4_K, 40, 2560, rng);
  CheckEmbed(k::WeightFormat::kQ6_K, 17, 512, rng);

  // A geometry the kernel is not written for is refused, not approximated.
  bool refused = false;
  try {
    k::EmbedArgs bad{};
    std::vector<std::int32_t> tokens{0};
    std::vector<float> out(200);
    const std::uint8_t* table = Device(std::vector<std::uint8_t>(4096, 0));
    bad.table = table;
    bad.format = k::WeightFormat::kQ6_K;
    bad.tokens = Device(tokens);
    bad.y = Device(out);
    bad.rows = 1;
    bad.n = 256;  // one super-block: the row would not end on a word
    k::EmbedRows(bad, nullptr);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  std::cout << (refused ? "PASS " : "FAIL ")
            << "a Q6_K table one super-block wide is refused\n";
  if (!refused) {
    ++failures;
  }

  bool accepted = false;
  try {
    k::EmbedArgs ok{};
    std::vector<std::int32_t> tokens{0};
    std::vector<float> out(512);
    ok.table = Device(std::vector<std::uint8_t>(1024, 0));
    ok.format = k::WeightFormat::kQ4_K;
    ok.tokens = Device(tokens);
    ok.y = Device(out);
    ok.rows = 1;
    ok.n = 256;
    k::EmbedRows(ok, nullptr);
    HIP_CHECK(hipStreamSynchronize(nullptr));
    accepted = true;
  } catch (const std::invalid_argument&) {
    accepted = false;
  }
  std::cout << (accepted ? "PASS " : "FAIL ")
            << "a Q4_K table one super-block wide is gathered\n";
  if (!accepted) {
    ++failures;
  }
  FreeAll();

  // The grouped launcher says no rather than launch a shape it cannot do: an
  // activation off the boundary the loads assume, a width the super-blocks do
  // not divide, a Q6_K row that would end mid-word, and more rows than a
  // group's slot list can hold.
  struct Refusal {
    const char* name;
    k::WeightFormat format;
    std::uint32_t rows;
    std::uint32_t kdim;
    bool misaligned;
  };
  const std::vector<Refusal> refusals{
      {"an activation off a sixteen-byte boundary", k::WeightFormat::kQ4_K, 2,
       512, true},
      {"a width the super-blocks do not divide", k::WeightFormat::kQ4_K, 2, 384,
       false},
      {"a Q6_K row of one super-block", k::WeightFormat::kQ6_K, 2, 256, false},
      {"a batch wider than a group's slot list", k::WeightFormat::kQ4_K,
       k::kMaxGroupSlots + 1, 512, false},
  };
  for (const Refusal& r : refusals) {
    const Built w = Build(r.format, 1, 8, 512, rng);
    std::vector<float> x((k::kMaxGroupSlots + 1) * 512, 0.0F),
        y((k::kMaxGroupSlots + 2) * 8, kPoison);
    k::GroupedGemvArgs args{};
    args.format = r.format;
    args.max_groups = 1;
    args.slots_per_row = 1;
    args.rows = r.rows;
    args.m = 8;
    args.k = r.kdim;
    args.groups = Device(MakeTable({{0, {0, 1}}}));
    args.y = Device(y);
    args.w = r.misaligned ? static_cast<const void*>(RawDevice(4096) + 4)
                          : static_cast<const void*>(Device(w.bytes));
    args.x = r.misaligned ? reinterpret_cast<const float*>(RawDevice(8192) + 4)
                          : Device(x);
    const bool refused_launch = !k::GroupedGemv(args, nullptr);
    std::cout << (refused_launch ? "PASS " : "FAIL ") << r.name << "\n";
    if (!refused_launch) {
      ++failures;
    }
    FreeAll();
  }

  if (std::getenv("KOLIBRI_TIME") != nullptr) {
    // Decode width: one row over six experts, the launch bound once to the six
    // groups a row can hold and once to every expert of the model. The pair
    // says what the blocks that find no group of theirs are worth.
    const std::vector<Group> six{{0, {0}},   {37, {1}},  {91, {2}},
                                 {140, {3}}, {255, {4}}, {383, {5}}};
    for (const k::WeightFormat f : formats) {
      const char* fmt = f == k::WeightFormat::kQ4_K ? "Q4_K" : "Q6_K";
      char name[96];
      std::snprintf(name, sizeof(name), "%s expert 512x2560 bound 6", fmt);
      Timed(name, f, 384, 512, 2560, 1, 6, six, 6, rng);
      std::snprintf(name, sizeof(name), "%s expert 512x2560 bound 384", fmt);
      Timed(name, f, 384, 512, 2560, 1, 6, six, 384, rng);
      std::snprintf(name, sizeof(name), "%s expert 2560x512 bound 6", fmt);
      Timed(name, f, 384, 2560, 512, 1, 6, six, 6, rng);
      std::snprintf(name, sizeof(name), "%s expert 2560x512 bound 384", fmt);
      Timed(name, f, 384, 2560, 512, 1, 6, six, 384, rng);
    }

    // The two shapes the mixture can be given at the same number of slots: one
    // table over a hundred and twenty-eight rows, where an expert appears once
    // and its weights pay once, and the same slots cut into the tables a
    // sixteen-row step would produce, where the experts a hot row chose are
    // read again in every piece. Seven hundred and sixty-eight slots over
    // three hundred and eighty-four experts, two slots each, which is the
    // average a hundred and twenty-eight rows of this model produce, against
    // ninety-six slots over ninety-six experts - and eight of the latter make
    // the width the one table covers.
    std::vector<Group> whole;
    for (std::int32_t e = 0; e < 384; ++e) {
      whole.push_back({e, {2 * e, 2 * e + 1}});
    }
    std::vector<Group> stepped;
    for (std::int32_t s = 0; s < 96; ++s) {
      stepped.push_back({s, {s}});
    }
    for (const k::WeightFormat f : formats) {
      const char* fmt = f == k::WeightFormat::kQ4_K ? "Q4_K" : "Q6_K";
      char name[96];
      std::snprintf(name, sizeof(name), "%s expert 512x2560 16r bound 96 x8",
                    fmt);
      Timed(name, f, 384, 512, 2560, 16, 6, stepped, 96, rng);
      std::snprintf(name, sizeof(name), "%s expert 512x2560 128r bound 384",
                    fmt);
      Timed(name, f, 384, 512, 2560, 128, 6, whole, 384, rng);
      std::snprintf(name, sizeof(name), "%s expert 2560x512 16r bound 96 x8",
                    fmt);
      Timed(name, f, 384, 2560, 512, 16, 6, stepped, 96, rng);
      std::snprintf(name, sizeof(name), "%s expert 2560x512 128r bound 384",
                    fmt);
      Timed(name, f, 384, 2560, 512, 128, 6, whole, 384, rng);
    }
  }

  if (failures != 0) {
    std::cout << failures << " gemv checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 grouped projection matches the FP64 dot over every "
               "case\n";
  return 0;
}
