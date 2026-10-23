// Qwen3.5-MoE's router against an FP64 formula over the same float inputs,
// so the only difference under test is the kernel's own arithmetic. The
// oracle rounds each logit to float because the contract publishes it as
// float and the weights are formed from what a verifier can read back.
//
// What the cases pin: the top-k order is the order of the logits (the softmax
// is monotone), a tie keeps the lower expert index, the weights are the
// chosen logits' softmax renormalised to sum 1, the group table names the
// experts that were used in increasing order with their slots in increasing
// order behind a leading count and says nothing at all about the experts
// nobody chose, counts add up to rows * used, and a row run alone gets bit
// for bit the ids, weights and logits of the same row inside a batch.
//
// Cases that need a logit of a known value build it rather than hope for it
// out of random data: an expert whose row of the router is zero but for one
// 1.0 reads a single element of x, so its logit is exactly that float and the
// gap between two candidates is exactly the difference of two floats. The
// random router rows of such fixtures are scaled to a tenth of a logit, which
// keeps them clear of the crafted band.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"

namespace k = gufo::models::qwen35moe::rocm;

namespace {

int failures = 0;

constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kExperts = 256;
constexpr std::uint32_t kUsed = 8;
constexpr std::int32_t kPoisonId = -12345;

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

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

struct Fixture {
  std::uint32_t rows = 1;
  std::uint32_t hidden = kHidden;
  std::uint32_t experts = kExperts;
  std::uint32_t used = kUsed;
  std::vector<float> x;       ///< [rows][hidden]
  std::vector<float> router;  ///< [experts][hidden]
};

Fixture RandomFixture(std::uint32_t rows, std::uint32_t hidden,
                      std::uint32_t experts, double router_scale,
                      std::uint32_t seed, std::uint32_t used = kUsed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  Fixture f;
  f.rows = rows;
  f.hidden = hidden;
  f.experts = experts;
  f.used = used;
  f.x.resize(static_cast<std::size_t>(rows) * hidden);
  f.router.resize(static_cast<std::size_t>(experts) * hidden);
  for (float& v : f.x) {
    v = normal(rng);
  }
  for (float& v : f.router) {
    v = static_cast<float>(normal(rng) * router_scale);
  }
  return f;
}

constexpr std::uint32_t kAnyRow = 0xFFFFFFFFU;

/// One expert whose logit the test dictates. Its row of the router becomes
/// zero but for a single 1.0 at a fresh element of x, which every row of the
/// batch then fills with `logit` or holds at zero.
struct Craft {
  std::uint32_t expert;
  std::uint32_t row = kAnyRow;
  float logit = 0.0F;
};

void CraftLogits(Fixture* f, const std::vector<Craft>& crafts) {
  std::uint32_t h = 0;
  for (const Craft& c : crafts) {
    float* w =
        f->router.data() + static_cast<std::size_t>(c.expert) * f->hidden;
    for (std::uint32_t i = 0; i < f->hidden; ++i) {
      w[i] = 0.0F;
    }
    w[h] = 1.0F;
    for (std::uint32_t r = 0; r < f->rows; ++r) {
      f->x[static_cast<std::size_t>(r) * f->hidden + h] =
          (c.row == kAnyRow || c.row == r) ? c.logit : 0.0F;
    }
    ++h;
  }
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

struct Expected {
  std::vector<float> logits;  ///< [rows][experts], rounded to float
  std::vector<std::int32_t> ids;     ///< [rows][used]
  std::vector<float> weights;        ///< [rows][used], sum to 1
};

Expected Oracle(const Fixture& f) {
  Expected e;
  e.logits.resize(static_cast<std::size_t>(f.rows) * f.experts);
  e.ids.resize(static_cast<std::size_t>(f.rows) * f.used);
  e.weights.resize(static_cast<std::size_t>(f.rows) * f.used);
  for (std::uint32_t r = 0; r < f.rows; ++r) {
    // Logits in double, published as float: the selection below reads what a
    // verifier can read back.
    for (std::uint32_t m = 0; m < f.experts; ++m) {
      double acc = 0.0;
      for (std::uint32_t i = 0; i < f.hidden; ++i) {
        acc += static_cast<double>(f.router[static_cast<std::size_t>(m) *
                                             f.hidden +
                                             i]) *
               f.x[static_cast<std::size_t>(r) * f.hidden + i];
      }
      e.logits[static_cast<std::size_t>(r) * f.experts + m] =
          static_cast<float>(acc);
    }
    // Top-k by (logit, low index), exactly the kernel's order.
    std::vector<std::uint32_t> order(f.experts);
    for (std::uint32_t m = 0; m < f.experts; ++m) {
      order[m] = m;
    }
    const float* row = e.logits.data() + static_cast<std::size_t>(r) *
                                              f.experts;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::uint32_t a, std::uint32_t b) {
                       return row[a] > row[b];
                     });
    double peak = -std::numeric_limits<double>::infinity();
    for (std::uint32_t j = 0; j < f.used; ++j) {
      peak = std::max(peak, static_cast<double>(row[order[j]]));
    }
    double total = 0.0;
    for (std::uint32_t j = 0; j < f.used; ++j) {
      total += std::exp(static_cast<double>(row[order[j]]) - peak);
    }
    for (std::uint32_t j = 0; j < f.used; ++j) {
      e.ids[static_cast<std::size_t>(r) * f.used + j] =
          static_cast<std::int32_t>(order[j]);
      e.weights[static_cast<std::size_t>(r) * f.used + j] = static_cast<float>(
          std::exp(static_cast<double>(row[order[j]]) - peak) / total);
    }
  }
  return e;
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

struct Got {
  std::vector<float> logits;
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  std::vector<std::uint32_t> counts;
  std::vector<std::int32_t> groups;
};

Got Run(const Fixture& f, bool want_logits, bool want_groups) {
  Got g;
  g.logits.resize(static_cast<std::size_t>(f.rows) * f.experts);
  g.ids.resize(static_cast<std::size_t>(f.rows) * f.used, kPoisonId);
  g.weights.resize(static_cast<std::size_t>(f.rows) * f.used);
  g.counts.resize(f.experts, 0);
  const std::uint32_t groups =
      std::min(f.experts, f.rows * f.used);
  g.groups.resize(k::GroupTableInts(groups), -1);
  float* dx = Device(f.x);
  float* drouter = Device(f.router);
  float* dlogits = Device(g.logits);
  std::int32_t* dids = Device(g.ids);
  float* dweights = Device(g.weights);
  std::uint32_t* dcounts = Device(g.counts);
  std::int32_t* dgroups = Device(g.groups);
  k::MoeRouteArgs args{};
  args.x = dx;
  args.router = drouter;
  args.logits = want_logits ? dlogits : nullptr;
  args.ids = dids;
  args.weights = dweights;
  args.counts = dcounts;
  args.groups = want_groups ? dgroups : nullptr;
  args.rows = f.rows;
  args.hidden = f.hidden;
  args.experts = f.experts;
  args.used = f.used;
  k::MoeRoute(args, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  g.logits = Read(dlogits, g.logits);
  g.ids = Read(dids, g.ids);
  g.weights = Read(dweights, g.weights);
  g.counts = Read(dcounts, g.counts);
  g.groups = Read(dgroups, g.groups);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(drouter));
  HIP_CHECK(hipFree(dlogits));
  HIP_CHECK(hipFree(dids));
  HIP_CHECK(hipFree(dweights));
  HIP_CHECK(hipFree(dcounts));
  HIP_CHECK(hipFree(dgroups));
  return g;
}

bool IdsEqual(const std::vector<std::int32_t>& a,
              const std::vector<std::int32_t>& b) {
  return a == b;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

void RandomRows() {
  Fixture f = RandomFixture(4, kHidden, kExperts, 0.1, 11);
  // Eight crafted experts with exact logits, half a logit apart, on top of a
  // random background a tenth as loud.
  std::vector<Craft> crafts;
  for (std::uint32_t j = 0; j < kUsed; ++j) {
    crafts.push_back({j, kAnyRow, 4.0F - static_cast<float>(j) * 0.5F});
  }
  CraftLogits(&f, crafts);
  Expected e = Oracle(f);
  Got g = Run(f, true, true);
  const bool ids_ok = IdsEqual(g.ids, e.ids);
  const double dw = MaxAbsDiff(g.weights, e.weights);
  double wsum = 0.0;
  for (float w : g.weights) {
    wsum += w;
  }
  wsum /= f.rows;
  const bool weights_ok = dw < 1e-6 && std::fabs(wsum - 1.0) < 1e-6;
  Report("route.random_rows", ids_ok && weights_ok,
         "ids " + std::string(ids_ok ? "match" : "MISMATCH") + " dw=" +
             Number(dw) + " wsum=" + Number(wsum));
}

void Ties() {
  Fixture f = RandomFixture(2, kHidden, kExperts, 0.05, 12);
  // Two experts with the same logit: the lower index wins the earlier slot.
  CraftLogits(&f, {{3, kAnyRow, 5.0F}, {7, kAnyRow, 5.0F}});
  Expected e = Oracle(f);
  Got g = Run(f, false, false);
  const bool ids_ok = IdsEqual(g.ids, e.ids);
  // The tied pair must come first, lower index first.
  const bool order_ok = g.ids[0] == 3 && g.ids[1] == 7;
  Report("route.ties", ids_ok && order_ok,
         "ids " + std::string(ids_ok ? "match" : "MISMATCH"));
}

void BatchInvariance() {
  Fixture f = RandomFixture(16, kHidden, kExperts, 0.1, 13);
  std::vector<Craft> crafts;
  for (std::uint32_t j = 0; j < kUsed; ++j) {
    crafts.push_back({j, kAnyRow, 4.0F - static_cast<float>(j) * 0.5F});
  }
  CraftLogits(&f, crafts);
  Got wide = Run(f, true, true);
  Fixture one = f;
  one.rows = 1;
  one.x.resize(one.hidden);
  Got narrow = Run(one, true, true);
  bool ok = true;
  for (std::uint32_t j = 0; j < f.used; ++j) {
    ok = ok && wide.ids[j] == narrow.ids[j];
    ok = ok && wide.weights[j] == narrow.weights[j];
  }
  for (std::uint32_t m = 0; m < f.experts; ++m) {
    ok = ok && wide.logits[m] == narrow.logits[m];
  }
  Report("route.batch_invariance", ok,
         "one row bit-identical solo and in sixteen");
}

void GroupTable(std::uint32_t rows) {
  Fixture f = RandomFixture(rows, kHidden, kExperts, 0.1, 14 + rows);
  std::vector<Craft> crafts;
  for (std::uint32_t j = 0; j < kUsed; ++j) {
    crafts.push_back({j, kAnyRow, 4.0F - static_cast<float>(j) * 0.5F});
  }
  CraftLogits(&f, crafts);
  Expected e = Oracle(f);
  Got g = Run(f, false, true);
  bool ok = IdsEqual(g.ids, e.ids);
  // Table structure: leading count, then per group expert/count/slots with
  // experts increasing and slots increasing, no empty group.
  const std::int32_t ngroups = g.groups[0];
  std::int32_t prev_expert = -1;
  std::size_t at = 1;
  std::uint32_t total_slots = 0;
  for (std::int32_t gi = 0; gi < ngroups && ok; ++gi) {
    const std::int32_t expert = g.groups[at];
    const std::int32_t n = g.groups[at + 1];
    ok = ok && expert > prev_expert && n > 0;
    prev_expert = expert;
    std::int32_t prev_slot = -1;
    for (std::int32_t s = 0; s < n && ok; ++s) {
      const std::int32_t slot = g.groups[at + 2 + s];
      ok = ok && slot > prev_slot;
      prev_slot = slot;
      // The slot's row chose this expert.
      ok = ok && g.ids[slot] == expert;
      ++total_slots;
    }
    at += k::kGroupInts;
  }
  ok = ok && total_slots == f.rows * f.used;
  std::uint32_t count_sum = 0;
  for (std::uint32_t c : g.counts) {
    count_sum += c;
  }
  ok = ok && count_sum == f.rows * f.used;
  Report(("route.group_table_" + std::to_string(rows)).c_str(), ok,
         "groups=" + std::to_string(ngroups) + " slots=" +
             std::to_string(total_slots));
}

void Refusals() {
  bool ok = true;
  Fixture f = RandomFixture(1, kHidden, kExperts, 0.1, 15);
  float* dx = Device(f.x);
  float* drouter = Device(f.router);
  std::vector<std::int32_t> ids(f.used);
  std::vector<float> weights(f.used);
  std::vector<std::uint32_t> counts(f.experts, 0);
  std::int32_t* dids = Device(ids);
  float* dweights = Device(weights);
  std::uint32_t* dcounts = Device(counts);
  k::MoeRouteArgs args{};
  args.x = dx;
  args.router = drouter;
  args.ids = dids;
  args.weights = dweights;
  args.counts = dcounts;
  args.rows = f.rows;
  args.hidden = f.hidden;
  args.experts = f.experts;
  args.used = f.used + 9;  // over kMaxUsed (16)
  try {
    k::MoeRoute(args, nullptr);
    ok = false;
  } catch (const std::invalid_argument&) {
  }
  args.used = f.used;
  args.hidden = f.hidden + 64;  // not a multiple of 128
  try {
    k::MoeRoute(args, nullptr);
    ok = false;
  } catch (const std::invalid_argument&) {
  }
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(drouter));
  HIP_CHECK(hipFree(dids));
  HIP_CHECK(hipFree(dweights));
  HIP_CHECK(hipFree(dcounts));
  Report("route.refusals", ok, "over-wide selection and width refused");
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  RandomRows();
  Ties();
  BatchInvariance();
  GroupTable(9);
  GroupTable(48);
  Refusals();

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
