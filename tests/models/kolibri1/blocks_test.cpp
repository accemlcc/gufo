// Kolibri 1's block arithmetic — the norms, the residual add, the expert
// activation and the end of a mixture block — against FP64 formulas over the
// same buffers, so the only difference under test is the kernel's own
// arithmetic.
//
// What the cases pin: eps inside the square root and after the mean rather than
// beside it, a norm with no weight tensor, the head-width and hidden-width rows,
// a row whose bits cannot change when a batch boundary moves it into another
// block, the sigmoid activation over arguments from -30 to 30 where a
// tanh-approximated gelu differs by a tenth, the mixture's contributions adding
// in expert order with the shared expert entering at weight 1 and nothing
// renormalizing the six, a count that ends mid-vector, and a row length that is
// not a multiple of 32.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"

namespace k = gufo::models::kolibri1::rocm;

namespace {

int failures = 0;

template <class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

std::vector<float> Read(const float* device, std::size_t count) {
  std::vector<float> host(count);
  HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host;
}

struct Score {
  double max_abs = 0.0;
  double mean_rel = 0.0;
  std::size_t worst_at = 0;
};

Score Compare(const std::vector<double>& want, const std::vector<float>& got) {
  Score s;
  double total = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - want[i]);
    total += diff / std::max(1e-3, std::abs(want[i]));
    if (diff > s.max_abs) {
      s.max_abs = diff;
      s.worst_at = i;
    }
  }
  s.mean_rel = total / static_cast<double>(want.size());
  return s;
}

void Check(const char* name, const char* what, const std::vector<double>& want,
           const std::vector<float>& got, double max_abs, double mean_rel) {
  const Score s = Compare(want, got);
  if (s.max_abs > max_abs || s.mean_rel > mean_rel) {
    ++failures;
    std::cout << "FAIL " << name << " " << what << ": max_abs=" << s.max_abs
              << " mean_rel=" << s.mean_rel << " (at " << want[s.worst_at]
              << ", got " << got[s.worst_at] << ")\n";
    return;
  }
  std::cout << "PASS " << name << " " << what << ": max_abs=" << s.max_abs
            << " mean_rel=" << s.mean_rel << "\n";
}

/// Exact equality of the stored bits, which is what an invariant of the kernel
/// rather than of the arithmetic looks like: the same row through two launches
/// of different width has to come out identical, not merely close.
void CheckBits(const char* name, const char* what,
               const std::vector<float>& alone,
               const std::vector<float>& batched, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (std::memcmp(&alone[i], &batched[i], sizeof(float)) != 0) {
      ++failures;
      std::cout << "FAIL " << name << " " << what << ": element " << i << " is "
                << alone[i] << " alone and " << batched[i] << " in a batch\n";
      return;
    }
  }
  std::cout << "PASS " << name << " " << what << "\n";
}

/// y = x * r * w over one row, with w absent for an unweighted norm. The eps is
/// inside the square root and after the mean, which is the placement every other
/// norm in this tree uses and the one a reshuffled kernel gets wrong.
void RmsRow(const std::vector<float>& x, std::size_t base,
            const std::vector<float>* w, std::uint32_t n, double eps,
            std::vector<double>* out) {
  double ss = 0.0;
  for (std::uint32_t d = 0; d < n; ++d) {
    const double v = x[base + d];
    ss += v * v;
  }
  const double r = 1.0 / std::sqrt(ss / static_cast<double>(n) + eps);
  for (std::uint32_t d = 0; d < n; ++d) {
    const double weight = w == nullptr ? 1.0 : static_cast<double>((*w)[d]);
    (*out)[base + d] = static_cast<double>(x[base + d]) * r * weight;
  }
}

struct NormCase {
  std::uint32_t rows;
  std::uint32_t n;
  bool weighted;
  double scale;  ///< magnitude of the row, small enough and eps takes over
  double eps;
  double max_abs;
  double mean_rel;
  const char* name;
};

void RunNorm(const NormCase& c, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  const std::size_t count = static_cast<std::size_t>(c.rows) * c.n;
  std::vector<float> x(count);
  for (float& v : x) {
    v = static_cast<float>(normal(rng) * c.scale);
  }
  std::vector<float> w(c.n);
  for (std::uint32_t d = 0; d < c.n; ++d) {
    w[d] = 0.5F + static_cast<float>(d % 11) / 10.0F;
  }
  float* dx = Device(x);
  float* dw = Device(w);
  float* dy = Device(std::vector<float>(count, 0.0F));
  float* dy_one = Device(std::vector<float>(count, -1.0F));

  const auto launch = [&](std::uint32_t rows, float* out) {
    k::NormRowsArgs a{};
    a.x = dx;
    a.w = c.weighted ? dw : nullptr;
    a.y = out;
    a.rows = rows;
    a.n = c.n;
    a.eps = static_cast<float>(c.eps);
    k::RmsNormRows(a, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
  };
  launch(c.rows, dy);
  const std::vector<float> got = Read(dy, count);

  std::vector<double> want(count);
  RmsRow(x, 0, c.weighted ? &w : nullptr, c.n, c.eps, &want);
  for (std::uint32_t r = 1; r < c.rows; ++r) {
    RmsRow(x, static_cast<std::size_t>(r) * c.n, c.weighted ? &w : nullptr, c.n,
           c.eps, &want);
  }
  Check(c.name, c.weighted ? "weighted norm" : "unweighted norm", want, got,
        c.max_abs, c.mean_rel);

  // The same first row through a launch of one row: the shape the kernel picks
  // may follow from n, but never from how many rows share the block grid.
  if (c.rows > 1) {
    launch(1, dy_one);
    const std::vector<float> alone = Read(dy_one, c.n);
    CheckBits(c.name, "a row alone matches the same row in a batch", alone,
              std::vector<float>(got.begin(), got.begin() + c.n), c.n);
  }
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dw),
                  static_cast<void*>(dy), static_cast<void*>(dy_one)}) {
    HIP_CHECK(hipFree(p));
  }
}

/// The add is one IEEE operation per element, so the expected result is the
/// correctly rounded one and the check asks for exactly that. The inputs are a
/// large magnitude and a small one so that nearly every sum needs rounding; with
/// values that add exactly, the case would pass on a kernel that dropped the
/// carry into the high bits.
void RunAdd(std::size_t count, std::mt19937& rng, const char* name) {
  std::uniform_real_distribution<float> big(-1.0e6F, 1.0e6F);
  std::uniform_real_distribution<float> small(-1.0e-3F, 1.0e-3F);
  std::vector<float> x(count);
  std::vector<float> add(count);
  for (std::size_t i = 0; i < count; ++i) {
    x[i] = big(rng);
    add[i] = i % 3 == 0 ? small(rng) : big(rng);
  }
  float* dx = Device(x);
  float* da = Device(add);
  k::AddRowsArgs a{};
  a.x = dx;
  a.add = da;
  a.count = count;
  k::AddRows(a, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const std::vector<float> got = Read(dx, count);
  std::vector<double> want(count);
  for (std::size_t i = 0; i < count; ++i) {
    // The double sum is exact for two floats, so rounding it once lands on the
    // single IEEE result.
    want[i] = static_cast<double>(
        static_cast<float>(static_cast<double>(x[i]) + static_cast<double>(add[i])));
  }
  Check(name, "residual add", want, got, 0.0, 0.0);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(da));
}

/// silu(gate) * up over a sweep of the whole knee, which is where the two
/// gated-linear activations of this family part: at gate = -3 the sigmoid form
/// is -0.142 and the tanh approximation is -0.002, so a kernel that picked the
/// wrong one cannot hide inside a tolerance.
void RunGateUp(std::size_t count, const char* name) {
  std::vector<float> gate(count);
  std::vector<float> up(count);
  for (std::size_t i = 0; i < count; ++i) {
    gate[i] = static_cast<float>(-30.0 +
                                 60.0 * static_cast<double>(i) /
                                     static_cast<double>(count - 1));
    up[i] = (i % 2 == 0 ? 1.0F : -1.0F) *
            (0.25F + static_cast<float>(i % 7) / 4.0F);
  }
  float* dg = Device(gate);
  float* du = Device(up);
  float* da = Device(std::vector<float>(count, 0.0F));
  k::GateUpArgs a{};
  a.gate = dg;
  a.up = du;
  a.act = da;
  a.count = count;
  k::GateUpAct(a, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  const std::vector<float> got = Read(da, count);
  std::vector<double> want(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double g = gate[i];
    want[i] = g / (1.0 + std::exp(-g)) * static_cast<double>(up[i]);
  }
  Check(name, "sigmoid gated linear unit", want, got, 5e-5, 1e-6);
  HIP_CHECK(hipFree(dg));
  HIP_CHECK(hipFree(du));
  HIP_CHECK(hipFree(da));
}

struct FinishCase {
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t used;
  double eps;
  const char* name;
};

void RunFinish(const FinishCase& c, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  const std::size_t routed_count =
      static_cast<std::size_t>(c.rows) * c.used * c.hidden;
  std::vector<float> routed(routed_count);
  for (float& v : routed) {
    v = normal(rng);
  }
  std::vector<float> shared(static_cast<std::size_t>(c.rows) * c.hidden);
  for (float& v : shared) {
    v = normal(rng) * 2.0F;
  }
  // Sigmoid weights that sum to about three: an implementation that
  // renormalized them over the chosen experts would shrink every row by that
  // factor, and one that dropped the shared expert would lose its share.
  std::vector<float> weights(static_cast<std::size_t>(c.rows) * c.used);
  for (float& v : weights) {
    v = 1.0F / (1.0F + std::exp(-normal(rng) * 2.0F));
  }
  std::vector<float> post_norm(c.hidden);
  for (std::uint32_t d = 0; d < c.hidden; ++d) {
    post_norm[d] = 0.6F + static_cast<float>(d % 5) / 5.0F;
  }
  std::vector<float> x(static_cast<std::size_t>(c.rows) * c.hidden);
  for (float& v : x) {
    v = normal(rng);
  }
  float* dr = Device(routed);
  float* ds = Device(shared);
  float* dw = Device(weights);
  float* dn = Device(post_norm);
  float* dx = Device(x);

  const auto launch = [&](std::uint32_t rows, float* out) {
    k::MoeFinishArgs a{};
    a.x = out;
    a.routed = dr;
    a.shared = ds;
    a.weights = dw;
    a.post_norm = dn;
    a.rows = rows;
    a.hidden = c.hidden;
    a.used = c.used;
    a.eps = static_cast<float>(c.eps);
    k::MoeFinish(a, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
  };
  launch(c.rows, dx);
  const std::vector<float> got = Read(dx, x.size());

  std::vector<double> want(x.size());
  for (std::uint32_t r = 0; r < c.rows; ++r) {
    double ss = 0.0;
    for (std::uint32_t d = 0; d < c.hidden; ++d) {
      double m = shared[static_cast<std::size_t>(r) * c.hidden + d];
      for (std::uint32_t j = 0; j < c.used; ++j) {
        const std::size_t slot = static_cast<std::size_t>(r) * c.used + j;
        m += static_cast<double>(weights[slot]) *
             static_cast<double>(routed[slot * c.hidden + d]);
      }
      ss += m * m;
    }
    const double n = 1.0 / std::sqrt(
                              ss / static_cast<double>(c.hidden) + c.eps);
    for (std::uint32_t d = 0; d < c.hidden; ++d) {
      double m = shared[static_cast<std::size_t>(r) * c.hidden + d];
      for (std::uint32_t j = 0; j < c.used; ++j) {
        const std::size_t slot = static_cast<std::size_t>(r) * c.used + j;
        m += static_cast<double>(weights[slot]) *
             static_cast<double>(routed[slot * c.hidden + d]);
      }
      const std::size_t i = static_cast<std::size_t>(r) * c.hidden + d;
      want[i] = static_cast<double>(x[i]) + m * n * static_cast<double>(post_norm[d]);
    }
  }
  Check(c.name, "mixture finish", want, got, 1e-5, 1e-6);

  if (c.rows > 1) {
    float* dx_one = Device(std::vector<float>(x));
    launch(1, dx_one);
    const std::vector<float> alone = Read(dx_one, c.hidden);
    CheckBits(c.name, "a token alone matches the same token in a batch", alone,
              std::vector<float>(got.begin(), got.begin() + c.hidden),
              c.hidden);
    HIP_CHECK(hipFree(dx_one));
  }
  for (void* p : {static_cast<void*>(dr), static_cast<void*>(ds),
                  static_cast<void*>(dw), static_cast<void*>(dn),
                  static_cast<void*>(dx)}) {
    HIP_CHECK(hipFree(p));
  }
}

/// True when the launcher refused the shape instead of launching it.
template <class Args, class Fn>
bool Refuses(const Args& a, Fn fn) {
  try {
    fn(a);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void ExpectRefused(bool refused, const char* what) {
  if (!refused) {
    ++failures;
  }
  std::cout << (refused ? "PASS " : "FAIL ") << what << "\n";
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

  // hidden and head_dim, the two widths this model norms.
  RunNorm({.rows = 5,
           .n = 2560,
           .weighted = true,
           .scale = 1.0,
           .eps = 1e-5,
           .max_abs = 1e-5,
           .mean_rel = 1e-6,
           .name = "hidden norm"},
          rng);
  RunNorm({.rows = 9,
           .n = 128,
           .weighted = true,
           .scale = 1.0,
           .eps = 1e-6,
           .max_abs = 1e-5,
           .mean_rel = 1e-6,
           .name = "head norm"},
          rng);
  RunNorm({.rows = 4,
           .n = 2560,
           .weighted = false,
           .scale = 1.0,
           .eps = 1e-5,
           .max_abs = 1e-5,
           .mean_rel = 1e-6,
           .name = "unweighted norm"},
          rng);
  // A row whose mean square is of the order of eps itself: wherever eps goes,
  // the scale moves by tens of percent, so no misplacement can pass.
  RunNorm({.rows = 6,
           .n = 2560,
           .weighted = true,
           .scale = 3e-3,
           .eps = 1e-5,
           .max_abs = 2e-6,
           .mean_rel = 1e-6,
           .name = "norm where eps dominates"},
          rng);
  RunNorm({.rows = 1,
           .n = 480,
           .weighted = true,
           .scale = 1.0,
           .eps = 1e-5,
           .max_abs = 1e-5,
           .mean_rel = 1e-6,
           .name = "row that is not a whole number of vectors per lane"},
          rng);

  RunAdd(3, rng, "three elements, no whole vector");
  RunAdd(2560 * 3 + 5, rng, "hidden rows plus a tail");
  RunAdd(100003, rng, "long flat count");

  RunGateUp(4096, "gate sweep from -30 to 30");
  RunGateUp(512 * 3 + 3, "expert intermediate plus a tail");

  RunFinish({.rows = 3, .hidden = 2560, .used = 6, .eps = 1e-5,
             .name = "mixture at hidden"},
            rng);
  RunFinish({.rows = 9, .hidden = 128, .used = 6, .eps = 1e-5,
             .name = "mixture at a narrow width"},
            rng);
  RunFinish({.rows = 2, .hidden = 2560, .used = 1, .eps = 1e-6,
             .name = "shared expert alone beside one routed expert"},
            rng);

  // Shapes the released model does not have are refused rather than rounded
  // into the nearest supported one.
  k::NormRowsArgs n{};
  n.rows = 1;
  n.x = reinterpret_cast<const float*>(0x1000);
  n.y = reinterpret_cast<float*>(0x2000);
  n.n = 100;
  n.eps = 1e-5F;
  ExpectRefused(Refuses(n, [](const k::NormRowsArgs& a) { k::RmsNormRows(a, nullptr); }),
                "a row length that is not a multiple of 32 is refused");
  n.n = 8320;
  ExpectRefused(Refuses(n, [](const k::NormRowsArgs& a) { k::RmsNormRows(a, nullptr); }),
                "a row wider than the kernel holds in registers is refused");

  k::MoeFinishArgs f{};
  f.rows = 1;
  f.hidden = 33;
  f.used = 6;
  f.eps = 1e-5F;
  ExpectRefused(Refuses(f, [](const k::MoeFinishArgs& a) { k::MoeFinish(a, nullptr); }),
                "a hidden size that is not a multiple of 32 is refused");

  float* base = nullptr;
  HIP_CHECK(hipMalloc(&base, 4096));
  k::AddRowsArgs add{};
  add.x = base + 1;
  add.add = base;
  add.count = 64;
  ExpectRefused(Refuses(add, [](const k::AddRowsArgs& a) { k::AddRows(a, nullptr); }),
                "an add whose base is not vector aligned is refused");
  k::GateUpArgs gu{};
  gu.gate = base;
  gu.up = base + 2;
  gu.act = base;
  gu.count = 64;
  ExpectRefused(Refuses(gu, [](const k::GateUpArgs& a) { k::GateUpAct(a, nullptr); }),
                "a gate or up base that is not vector aligned is refused");
  HIP_CHECK(hipFree(base));

  if (failures != 0) {
    std::cout << failures << " block checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 block arithmetic matches the reference over every case\n";
  return 0;
}
