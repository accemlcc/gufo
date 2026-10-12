// Kolibri 1's router against an FP64 formula over the same float inputs, so the
// only difference under test is the kernel's own arithmetic. The oracle rounds
// each logit to float because the contract publishes it as float and the weight
// is formed from what a verifier can read back.
//
// What the cases pin: the selection reads `logit + bias` while the weight is the
// sigmoid of the chosen expert's own unbiased logit, a tie in `logit + bias`
// keeps the lower expert index, a sixth and seventh candidate that differ by
// less than a float epsilon still order strictly instead of collapsing, the
// group table names the experts that were used in increasing order with their
// slots in increasing order behind a leading count and says nothing at all
// about the experts nobody chose, counts add up to rows * used, and a row run
// alone gets bit for bit the ids, weights and logits of the same row inside a
// batch of sixteen.
//
// Cases that need a logit of a known value build it rather than hope for it out
// of random data: an expert whose row of the router is zero but for one 1.0
// reads a single element of x, so its logit is exactly that float and the gap
// between two candidates is exactly the difference of two floats. The random
// router rows of such fixtures are scaled to a tenth of a logit, which keeps
// them clear of the crafted band.
//
// Needs a gfx1151 device; without one the check skips with 77.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"

namespace k = gufo::models::kolibri1::rocm;

namespace {

int failures = 0;

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kExperts = 384;
constexpr std::uint32_t kUsed = 6;
constexpr std::int32_t kPoisonId = -12345;
constexpr std::int32_t kPoisonInt = -7777;

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

template <class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

template <class T>
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
  std::vector<float> bias;    ///< [experts]
};

Fixture RandomFixture(std::uint32_t rows, std::uint32_t hidden,
                      std::uint32_t experts, double router_scale,
                      double bias_scale, std::uint32_t seed,
                      std::uint32_t used = kUsed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  Fixture f;
  f.rows = rows;
  f.hidden = hidden;
  f.experts = experts;
  f.used = used;
  f.x.resize(static_cast<std::size_t>(rows) * hidden);
  f.router.resize(static_cast<std::size_t>(experts) * hidden);
  f.bias.resize(experts);
  for (float& v : f.x) {
    v = normal(rng);
  }
  for (float& v : f.router) {
    v = static_cast<float>(normal(rng) * router_scale);
  }
  for (float& v : f.bias) {
    v = static_cast<float>(normal(rng) * bias_scale);
  }
  return f;
}

constexpr std::uint32_t kAnyRow = 0xFFFFFFFFU;

/// One expert whose logit the test dictates. Its row of the router becomes zero
/// but for a single 1.0 at a fresh element of x, which every row of the batch
/// then fills with `logit` or holds at zero.
struct Craft {
  std::uint32_t expert;
  std::uint32_t row = kAnyRow;
  float logit = 0.0F;
  float bias = 0.0F;
};

void CraftLogits(Fixture* f, const std::vector<Craft>& crafts) {
  std::uint32_t h = 0;
  for (const Craft& c : crafts) {
    float* w = f->router.data() + static_cast<std::size_t>(c.expert) * f->hidden;
    std::fill(w, w + f->hidden, 0.0F);
    w[h] = 1.0F;
    f->bias[c.expert] = c.bias;
    for (std::uint32_t r = 0; r < f->rows; ++r) {
      float* xr = f->x.data() + static_cast<std::size_t>(r) * f->hidden;
      xr[h] = (c.row == kAnyRow || c.row == r) ? c.logit : 0.0F;
    }
    ++h;
  }
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

float Sigmoid(double logit) {
  return static_cast<float>(1.0 / (1.0 + std::exp(-logit)));
}

/// One group of the expected table: the expert and the slots it owns. The
/// table itself carries them kGroupInts apart, so the expected groups are kept
/// apart too rather than as a flat stream of ints.
struct Group {
  std::int32_t expert;
  std::vector<std::int32_t> slots;
};

struct Want {
  std::vector<float> logits;  ///< [rows][experts] as the contract stores them
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  std::vector<std::uint32_t> counts;
  std::vector<Group> table;  ///< experts with a slot, increasing by expert
};

Want Oracle(const Fixture& f) {
  Want w;
  w.logits.resize(static_cast<std::size_t>(f.rows) * f.experts);
  for (std::uint32_t r = 0; r < f.rows; ++r) {
    const float* xr = f.x.data() + static_cast<std::size_t>(r) * f.hidden;
    for (std::uint32_t e = 0; e < f.experts; ++e) {
      const float* wr =
          f.router.data() + static_cast<std::size_t>(e) * f.hidden;
      double acc = 0.0;
      for (std::uint32_t h = 0; h < f.hidden; ++h) {
        acc += static_cast<double>(wr[h]) * static_cast<double>(xr[h]);
      }
      w.logits[static_cast<std::size_t>(r) * f.experts + e] =
          static_cast<float>(acc);
    }
  }
  w.ids.resize(static_cast<std::size_t>(f.rows) * f.used);
  w.weights.resize(w.ids.size());
  w.counts.assign(f.experts, 0);
  std::map<std::int32_t, std::vector<std::int32_t>> by_expert;
  for (std::uint32_t r = 0; r < f.rows; ++r) {
    const float* lr = w.logits.data() + static_cast<std::size_t>(r) * f.experts;
    std::vector<std::pair<double, std::int32_t>> score(f.experts);
    for (std::uint32_t e = 0; e < f.experts; ++e) {
      // Both addends are floats, so their sum in double is exact: a tie here is
      // a tie in any arithmetic that does not throw bits away.
      score[e] = {static_cast<double>(lr[e]) + static_cast<double>(f.bias[e]),
                  static_cast<std::int32_t>(e)};
    }
    std::sort(score.begin(), score.end(),
              [](const std::pair<double, std::int32_t>& a,
                 const std::pair<double, std::int32_t>& b) {
                return a.first != b.first ? a.first > b.first
                                          : a.second < b.second;
              });
    for (std::uint32_t j = 0; j < f.used; ++j) {
      const std::int32_t id = score[j].second;
      const std::size_t slot = static_cast<std::size_t>(r) * f.used + j;
      w.ids[slot] = id;
      w.weights[slot] = Sigmoid(static_cast<double>(lr[id]));
      ++w.counts[static_cast<std::size_t>(id)];
      by_expert[id].push_back(static_cast<std::int32_t>(slot));
    }
  }
  for (const auto& g : by_expert) {
    w.table.push_back({g.first, g.second});
  }
  return w;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

struct Got {
  std::vector<float> logits;
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  std::vector<std::uint32_t> counts;
  std::vector<std::int32_t> groups;
};

struct Ask {
  bool logits = true;
  bool counts = true;
  bool groups = true;
};

/// One launch, and what the buffers hold afterwards. Buffers the caller did not
/// ask for stay empty and the ones it did are poisoned first, so a kernel that
/// wrote nothing, or wrote past the end of its table, shows up instead of
/// looking plausible.
Got Launch(const Fixture& f, const Ask& ask, bool* threw = nullptr) {
  Got g;
  if (ask.logits) {
    g.logits.assign(static_cast<std::size_t>(f.rows) * f.experts, NAN);
  }
  g.ids.assign(static_cast<std::size_t>(f.rows) * f.used, kPoisonId);
  g.weights.assign(static_cast<std::size_t>(f.rows) * f.used, NAN);
  if (ask.counts) {
    g.counts.assign(f.experts, 0);  // the contract: zeroed by the caller
  }
  if (ask.groups) {
    g.groups.assign(k::GroupTableInts(f.rows * f.used), kPoisonInt);
  }

  float* dx = Device(f.x);
  float* dr = Device(f.router);
  float* db = Device(f.bias);
  float* dlogits = ask.logits ? Device(g.logits) : nullptr;
  std::int32_t* dids = Device(g.ids);
  float* dweights = Device(g.weights);
  std::uint32_t* dcounts = ask.counts ? Device(g.counts) : nullptr;
  std::int32_t* dgroups = ask.groups ? Device(g.groups) : nullptr;

  k::MoeRouteArgs a{};
  a.x = dx;
  a.router = dr;
  a.bias = db;
  a.logits = dlogits;
  a.ids = dids;
  a.weights = dweights;
  a.counts = dcounts;
  a.groups = dgroups;
  a.rows = f.rows;
  a.hidden = f.hidden;
  a.experts = f.experts;
  a.used = f.used;

  bool raised = false;
  try {
    k::MoeRoute(a, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
  } catch (const std::invalid_argument&) {
    raised = true;
  }
  if (threw != nullptr) {
    *threw = raised;
  }
  if (!raised) {
    if (ask.logits) {
      g.logits = Read(dlogits, std::move(g.logits));
    }
    g.ids = Read(dids, std::move(g.ids));
    g.weights = Read(dweights, std::move(g.weights));
    if (ask.counts) {
      g.counts = Read(dcounts, std::move(g.counts));
    }
    if (ask.groups) {
      g.groups = Read(dgroups, std::move(g.groups));
    }
  }
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dr),
                  static_cast<void*>(db), static_cast<void*>(dlogits),
                  static_cast<void*>(dids), static_cast<void*>(dweights),
                  static_cast<void*>(dcounts), static_cast<void*>(dgroups)}) {
    HIP_CHECK(hipFree(p));
  }
  return g;
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

/// Relative to the expected value where it has a magnitude worth speaking of,
/// absolute where it has not, so a logit that happens to land near zero cannot
/// turn rounding into a reported disaster.
double Error(const std::vector<float>& got, const std::vector<float>& want) {
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double diff = std::abs(static_cast<double>(got[i]) - want[i]);
    const double magnitude = std::abs(static_cast<double>(want[i]));
    worst = std::max(worst, diff / std::max(1e-3, magnitude));
  }
  return worst;
}

/// Largest absolute difference. A random row's logit is a sum of 2560 products
/// that cancel to a fraction of the terms, so its relative error reports where
/// the sum happened to land rather than how the arithmetic was done; the
/// absolute number is what a different accumulator moves.
double AbsError(const std::vector<float>& got,
                const std::vector<float>& want) {
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]));
  }
  return worst;
}

bool SameIds(const Got& g, const Want& w) {
  return g.ids == w.ids;
}

/// The chosen experts in the order the row got them, for a failure line that
/// says what differed.
std::string IdList(const std::vector<std::int32_t>& ids) {
  std::string text;
  for (const std::int32_t id : ids) {
    if (!text.empty()) {
      text += ',';
    }
    text += std::to_string(id);
  }
  return text;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// The two shapes the artifact runs: one row of decode and a full batch. The
/// gate is on the weights, the logits and the ids; the gap the selection turned
/// on is printed because a disagreement with a gap wider than this would mean
/// something other than the ordering failed.
void RandomRows(std::mt19937& rng) {
  for (const std::uint32_t rows : {1U, 16U}) {
    const Fixture f =
        RandomFixture(rows, kHidden, kExperts, 0.02, 0.5, rng());
    const Want w = Oracle(f);
    const Got g = Launch(f, Ask{});
    const char* tag =
        rows == 1 ? "random decode row" : "random batch of sixteen rows";
    const double weight_error = Error(g.weights, w.weights);
    const double logit_error = AbsError(g.logits, w.logits);
    // The weight is the sigmoid of the logit as stored, and stored it is in
    // float32: `dw / w = (1 - w) * dl <= dl`, so a weight cannot be relatively
    // wrong by more than its logit is wrong in absolute terms, and one unit in
    // the last place of a logit of order one is 1.2e-7. Two of those is the
    // gate: it lets the kernel's own rounding through and nothing coarser,
    // because a half-precision accumulator or a dot over part of the row lands
    // orders away. The crafted rows are exact to the bit for the same reason —
    // their logits are a single product and carry no chain at all.
    Report(tag, weight_error < 2e-7 && logit_error < 1e-6 && SameIds(g, w),
           "weight error " + Number(weight_error) + " relative, ids " +
               std::string(SameIds(g, w) ? "exact" : "differ"));

    double narrowest = 1e30;
    for (std::uint32_t r = 0; r < rows; ++r) {
      std::vector<double> score(f.experts);
      for (std::uint32_t e = 0; e < f.experts; ++e) {
        score[e] = static_cast<double>(
                       w.logits[static_cast<std::size_t>(r) * f.experts + e]) +
                   static_cast<double>(f.bias[e]);
      }
      std::sort(score.begin(), score.end(), std::greater<double>());
      narrowest = std::min(narrowest, score[f.used - 1] - score[f.used]);
    }
    std::cout << "     " << tag << ": narrowest sixth to seventh gap "
              << Number(narrowest) << ", weight range ["
              << Number(*std::min_element(g.weights.begin(), g.weights.end()))
              << ", " << Number(*std::max_element(g.weights.begin(),
                                                  g.weights.end()))
              << "], logits " << Number(logit_error) << " max absolute\n";
  }
}

/// (a) A bias that moves the sixth expert out and another one in and moves no
/// weight at all: the five experts that keep their places keep their bits, and
/// the newcomer arrives with the sigmoid of its own logit, not of its score.
void BiasMovesSelectionOnly(std::mt19937& rng) {
  Fixture f = RandomFixture(3, kHidden, kExperts, 0.002, 0.0, rng());
  CraftLogits(&f, {{11, kAnyRow, 3.0F, 0.0F},
                   {21, kAnyRow, 2.9F, 0.0F},
                   {31, kAnyRow, 2.8F, 0.0F},
                   {41, kAnyRow, 2.7F, 0.0F},
                   {51, kAnyRow, 2.6F, 0.0F},
                   {61, kAnyRow, 2.5F, 0.0F},
                   {71, kAnyRow, 2.4F, 0.0F}});
  const Want without = Oracle(f);
  const Got plain = Launch(f, Ask{});

  // Expert 71 is seventh by 0.1, so a bias of 0.15 lifts it past 61 and past
  // nobody else: exactly one slot changes hands.
  f.bias[71] = 0.15F;
  const Want with = Oracle(f);
  const Got biased = Launch(f, Ask{});

  bool changed = false;
  bool head_kept = true;
  for (std::size_t i = 0; i < with.ids.size(); ++i) {
    changed |= with.ids[i] != without.ids[i];
  }
  for (std::uint32_t r = 0; r < f.rows; ++r) {
    for (std::uint32_t j = 0; j + 1 < f.used; ++j) {
      const std::size_t s = static_cast<std::size_t>(r) * f.used + j;
      head_kept &= biased.ids[s] == plain.ids[s] &&
                   biased.weights[s] == plain.weights[s];
    }
  }
  const double weight_error = Error(biased.weights, with.weights);
  Report("a bias that changes the selection changes no weight",
         changed && head_kept && weight_error < 1e-7 && SameIds(biased, with),
         "selection changed, five kept experts bit for bit identical, weight "
         "error " +
             Number(weight_error));

  // How far the case would have been off had the bias reached the weight. A
  // check that cannot fail is not a check, so the distance is measured.
  double leak = 0.0;
  for (std::size_t i = 0; i < with.ids.size(); ++i) {
    const std::uint32_t r = static_cast<std::uint32_t>(i / f.used);
    const std::int32_t e = with.ids[i];
    if (f.bias[e] == 0.0F) {
      continue;  // only an expert the bias moved can show a leak
    }
    const double l = static_cast<double>(
        with.logits[static_cast<std::size_t>(r) * f.experts + e]);
    leak = std::max(
        leak, std::abs(static_cast<double>(biased.weights[i]) -
                       Sigmoid(l + static_cast<double>(f.bias[e]))));
  }
  Report("the weight is the unbiased sigmoid, not the one the bias would give",
         leak > 1e-4 && weight_error < 1e-7,
         "closest weight " + Number(leak) + " away from a bias-carrying one");
}

/// (b) An exact tie in `logit + bias`: once from a clone of a router row and an
/// equal bias, so the two scores are the same bits, and once from two different
/// splits of the same sum, where the winner carries the bias and its weight has
/// to be read from its own logit.
void Ties(std::mt19937& rng) {
  {
    Fixture f = RandomFixture(1, kHidden, kExperts, 0.02, 0.5, rng());
    const Want before = Oracle(f);
    const std::int32_t sixth = before.ids[f.used - 1];
    std::int32_t lower = -1;
    for (std::int32_t e = sixth - 1; e >= 0; --e) {
      if (std::find(before.ids.begin(), before.ids.end(), e) ==
          before.ids.end()) {
        lower = e;
        break;
      }
    }
    if (lower < 0) {
      Report("an exact tie keeps the lower expert index", false,
             "no unchosen expert below the sixth to clone onto");
    } else {
      std::copy_n(
          f.router.data() + static_cast<std::size_t>(sixth) * f.hidden,
          f.hidden,
          f.router.data() + static_cast<std::size_t>(lower) * f.hidden);
      f.bias[lower] = f.bias[sixth];
      const Want w = Oracle(f);
      const Got g = Launch(f, Ask{});
      const bool swapped =
          std::find(g.ids.begin(), g.ids.end(), lower) != g.ids.end() &&
          std::find(g.ids.begin(), g.ids.end(), sixth) == g.ids.end();
      Report("an exact tie keeps the lower expert index",
             swapped && SameIds(g, w) && Error(g.weights, w.weights) < 1e-7,
             "expert " + std::to_string(lower) + " took the sixth slot from " +
                 std::to_string(sixth) + " on equal scores");
    }
  }
  {
    Fixture f = RandomFixture(2, kHidden, kExperts, 0.002, 0.0, rng());
    CraftLogits(&f, {{10, kAnyRow, 5.0F, 0.0F},
                     {20, kAnyRow, 4.5F, 0.0F},
                     {30, kAnyRow, 4.0F, 0.0F},
                     {40, kAnyRow, 3.5F, 0.0F},
                     {50, kAnyRow, 3.0F, 0.0F},
                     {250, kAnyRow, 2.25F, 0.0F},  // score 2.25
                     {100, kAnyRow, 2.0F, 0.25F},  // score 2.25, lower index
                     {60, kAnyRow, 2.0F, 0.0F}});  // score 2.0, next out
    const Want w = Oracle(f);
    const Got g = Launch(f, Ask{});
    bool expected = true;
    for (std::uint32_t r = 0; r < f.rows; ++r) {
      const std::int32_t want_ids[kUsed] = {10, 20, 30, 40, 50, 100};
      const std::int32_t* got_ids = g.ids.data() + r * kUsed;
      expected &= std::equal(want_ids, want_ids + kUsed, got_ids);
      // The winner's weight is sigmoid(2.0). A bias that leaked into it, or the
      // score mistaken for the logit, would give sigmoid(2.25) instead.
      const double sixth_weight = static_cast<double>(g.weights[r * kUsed + (kUsed - 1)]);
      expected &= std::abs(sixth_weight - Sigmoid(2.0)) < 1e-7;
      expected &= std::abs(sixth_weight - Sigmoid(2.25)) > 1e-3;
    }
    Report("two different splits of one tied score leave the lower index first",
           expected && SameIds(g, w) && Error(g.weights, w.weights) < 1e-7,
           "ids[5]=100 with weight " + Number(Sigmoid(2.0)) +
               ", the sigmoid of its logit and not of its score, while 250 is "
               "seventh");
  }
}

/// (c) The sixth and the seventh candidate less than a float epsilon apart,
/// with the lower index on the losing side: an order with any slack in it would
/// hand the slot to the wrong expert.
void SubEpsilonBoundary(std::mt19937& rng) {
  const float just_under = std::nextafterf(1.0F, 0.0F);
  Fixture f = RandomFixture(2, kHidden, kExperts, 0.002, 0.0, rng());
  CraftLogits(&f, {{10, kAnyRow, 5.0F, 0.0F},
                   {20, kAnyRow, 4.5F, 0.0F},
                   {30, kAnyRow, 4.0F, 0.0F},
                   {40, kAnyRow, 3.5F, 0.0F},
                   {50, kAnyRow, 3.0F, 0.0F},
                   {300, kAnyRow, 1.0F, 0.0F},
                   // One ulp of 1.0 below, which is half a float epsilon, and
                   // helped by a bias too small to close even that.
                   {12, kAnyRow, just_under, 1e-9F}});
  const Want w = Oracle(f);
  const Got g = Launch(f, Ask{});
  const double gap = 1.0 - (static_cast<double>(just_under) + 1e-9);
  bool expected = gap > 0.0 && gap < std::numeric_limits<float>::epsilon();
  for (std::uint32_t r = 0; r < f.rows; ++r) {
    expected &= g.ids[r * kUsed + (kUsed - 1)] == 300;
    const std::int32_t* row = g.ids.data() + r * kUsed;
    expected &= std::find(row, row + kUsed, 12) == row + kUsed;
  }
  Report("a sixth and seventh candidate under a float epsilon apart still order",
         expected && SameIds(g, w) && Error(g.weights, w.weights) < 1e-7,
         "gap " + Number(gap) + " against an epsilon of " +
             Number(std::numeric_limits<float>::epsilon()) +
             ", sixth is 300 and 12 is out");
}

/// (d) The group table: a leading count, the experts that were used in
/// increasing order, their slots in increasing order, and no word spent on the
/// experts nobody chose. Five experts serve every row of the batch and four
/// serve one row each, so both shapes of group appear, and two of the five share
/// a score so that a tie cannot reorder the table into something the count
/// disagrees with.
void GroupTable(std::mt19937& rng) {
  constexpr std::uint32_t rows = 9;
  Fixture f = RandomFixture(rows, kHidden, kExperts, 0.002, 0.0, rng());
  std::vector<Craft> crafts = {{10, kAnyRow, 5.0F, 0.0F},
                               {20, kAnyRow, 4.5F, 0.0F},
                               {30, kAnyRow, 4.0F, 0.0F},
                               {40, kAnyRow, 4.0F, 0.0F},
                               {50, kAnyRow, 3.5F, 0.0F},
                               {60, kAnyRow, 3.0F, 0.0F}};
  for (std::uint32_t r = 0; r < rows; ++r) {
    crafts.push_back(
        {70U + r, r, 4.25F + 0.001F * static_cast<float>(r), 0.0F});
  }
  CraftLogits(&f, crafts);
  const Want w = Oracle(f);
  const Got g = Launch(f, Ask{});

  const std::int32_t count = g.groups.empty() ? -1 : g.groups[0];
  bool structure = count > 0;
  bool fits = true;
  bool matches = count == static_cast<std::int32_t>(w.table.size());
  std::uint32_t listed = 0;
  std::int32_t widest = 0;
  std::int32_t previous = -1;
  std::vector<char> seen(f.experts, 0);
  for (std::int32_t i = 0; i < count && structure; ++i) {
    const std::int32_t* entry = g.groups.data() + 1 + i * k::kGroupInts;
    structure &= entry[0] > previous && entry[1] > 0 &&
                 entry[0] < static_cast<std::int32_t>(f.experts);
    previous = entry[0];
    seen[static_cast<std::size_t>(entry[0])] = 1;
    for (std::int32_t s = 1; s < entry[1]; ++s) {
      structure &= entry[2 + s] > entry[1 + s];
    }
    if (i >= static_cast<std::int32_t>(w.table.size())) {
      matches = false;
    } else {
      const Group& want = w.table[static_cast<std::size_t>(i)];
      matches &= entry[0] == want.expert &&
                 entry[1] == static_cast<std::int32_t>(want.slots.size());
      for (std::int32_t s = 0; s < entry[1]; ++s) {
        matches &= entry[2 + s] == want.slots[static_cast<std::size_t>(s)];
      }
    }
    widest = std::max(widest, entry[1]);
    listed += static_cast<std::uint32_t>(entry[1]);
    fits &= static_cast<std::uint32_t>(entry[1]) <= k::kMaxGroupSlots;
  }
  // The table is exactly as long as its own count says.
  for (std::size_t i = 1 + static_cast<std::size_t>(count) * k::kGroupInts;
       i < g.groups.size(); ++i) {
    structure &= g.groups[i] == kPoisonInt;
  }
  // And an expert with no assignment is absent from it, which only the count
  // and the membership together can say.
  bool membership = true;
  for (std::uint32_t e = 0; e < f.experts; ++e) {
    membership &= seen[e] == (w.counts[e] != 0 ? 1 : 0);
  }
  // Five crafted experts serve every row of the batch and one serves each row
  // alone, so exactly fourteen experts appear.
  constexpr std::int32_t kGroupsInTable = 5 + static_cast<std::int32_t>(rows);
  Report("the group table lists the used experts in order with their slots",
         structure && matches && fits && membership &&
             listed == rows * kUsed && count == kGroupsInTable,
         "count=" + std::to_string(count) + " slots=" + std::to_string(listed) +
             " of " + std::to_string(rows * kUsed) + ", widest group " +
             std::to_string(widest) + ", the other " +
             std::to_string(f.experts - count) + " experts unnamed");
}

/// (e) The per-expert counts add up to one per slot and land where the oracle
/// says they land.
void Counts(std::mt19937& rng) {
  constexpr std::uint32_t rows = 16;
  const Fixture f = RandomFixture(rows, kHidden, kExperts, 0.02, 0.5, rng());
  const Want w = Oracle(f);
  const Got g = Launch(f, Ask{});
  std::uint64_t total = 0;
  std::uint32_t named = 0;
  for (std::uint32_t e = 0; e < f.experts; ++e) {
    total += g.counts[e];
    named += w.counts[e] != 0 ? 1 : 0;
  }
  Report("counts hold one assignment per slot",
         g.counts == w.counts && total == rows * static_cast<std::uint64_t>(kUsed),
         "total " + std::to_string(total) + " over " + std::to_string(named) +
             " experts, largest " +
             std::to_string(*std::max_element(g.counts.begin(), g.counts.end())));
}

/// (f) The same row alone and inside the widest batch the table can hold, for a
/// random row and for a row sitting on the sub-epsilon boundary.
void BatchInvariance(std::mt19937& rng) {
  constexpr std::uint32_t kShared = 7;
  for (int which = 0; which < 2; ++which) {
    Fixture batch = which == 0
                        ? RandomFixture(16, kHidden, kExperts, 0.02, 0.5, rng())
                        : RandomFixture(16, kHidden, kExperts, 0.002, 0.0, rng());
    if (which == 1) {
      CraftLogits(&batch, {{10, kAnyRow, 5.0F, 0.0F},
                           {20, kAnyRow, 4.5F, 0.0F},
                           {30, kAnyRow, 4.0F, 0.0F},
                           {40, kAnyRow, 3.5F, 0.0F},
                           {50, kAnyRow, 3.0F, 0.0F},
                           {300, kAnyRow, 1.0F, 0.0F},
                           {12, kAnyRow, std::nextafterf(1.0F, 0.0F), 1e-9F}});
    }
    Fixture alone = batch;
    alone.rows = 1;
    alone.x.assign(
        batch.x.begin() + static_cast<std::ptrdiff_t>(kShared) * kHidden,
        batch.x.begin() + static_cast<std::ptrdiff_t>(kShared + 1) * kHidden);

    const Got wide = Launch(batch, Ask{});
    const Got single = Launch(alone, Ask{});
    bool ids_same = true;
    bool weights_same = true;
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      // Bit for bit rather than within a tolerance: a batch boundary must not
      // change a token's numbers at all.
      ids_same &= wide.ids[kShared * kUsed + j] == single.ids[j];
      weights_same &= wide.weights[kShared * kUsed + j] == single.weights[j];
    }
    bool logits_same = true;
    for (std::uint32_t e = 0; e < batch.experts; ++e) {
      logits_same &=
          wide.logits[static_cast<std::size_t>(kShared) * batch.experts + e] ==
          single.logits[e];
    }
    Report(which == 0
               ? "a row alone gets the ids and weights of the same row in a batch"
               : "a row on a sub-epsilon boundary survives a batch boundary",
           ids_same && weights_same && logits_same,
           std::string(which == 0 ? "random" : "crafted") + " row " +
               std::to_string(kShared) + ": six ids, six weights and " +
               std::to_string(batch.experts) + " logits identical bit for bit");
  }
}

/// A shape outside the artifact's, where an expert is narrower than a wave and
/// every expert of the row is chosen, and the outputs a caller may leave out.
/// The gates are the ones of the random fixtures: the weight of a slot may be
/// off by as much as the logit it comes from, and that is a unit in the last
/// place of a float32, which this shape reaches with only 128 products to sum.
void SmallShapeAndOptionalOutputs(std::mt19937& rng) {
  const Fixture f = RandomFixture(3, 128, 8, 0.1, 1.0, rng(), 8);
  const Want w = Oracle(f);
  const Got bare =
      Launch(f, Ask{.logits = false, .counts = false, .groups = false});
  Report("eight experts below a wave, all of them chosen",
         SameIds(bare, w) && Error(bare.weights, w.weights) < 2e-7,
         "ids and weights only, no logits, no counts, no table: ids " +
             std::string(SameIds(bare, w) ? "exact"
                                          : "kernel " + IdList(bare.ids) +
                                                " oracle " + IdList(w.ids)) +
             ", weight error " + Number(Error(bare.weights, w.weights)));

  // With nothing published the kernel needs logits of its own, and it has to
  // select on exactly the numbers the other hand-over would have shown.
  const Got full = Launch(f, Ask{});
  Report("the logits the kernel keeps to itself select the same experts",
         SameIds(full, w) && Error(full.weights, w.weights) < 2e-7 &&
             AbsError(full.logits, w.logits) < 1e-6,
         "eight logits published: ids " +
             std::string(SameIds(full, w) ? "exact"
                                          : "kernel " + IdList(full.ids) +
                                                " oracle " + IdList(w.ids)) +
             ", weight error " + Number(Error(full.weights, w.weights)));
}

/// Shapes the model does not have are refused before anything is launched,
/// which the poisoned buffers show.
void Refusals() {
  struct Bad {
    std::uint32_t rows;
    std::uint32_t hidden;
    std::uint32_t experts;
    std::uint32_t used;
    const char* name;
  };
  const Bad bad[] = {
      {17, kHidden, kExperts, kUsed, "seventeen rows"},
      {4, kHidden, kExperts, kExperts + 1,
       "more experts used per row than the model has"},
      {4, 2500, kExperts, kUsed,
       "a hidden size that is not a multiple of 128"},
  };
  for (const Bad& b : bad) {
    const Fixture f = RandomFixture(b.rows, b.hidden, b.experts, 0.02, 0.5,
                                    991U, b.used);
    bool threw = false;
    const Got g = Launch(f, Ask{}, &threw);
    bool untouched = true;
    for (std::int32_t v : g.ids) {
      untouched &= v == kPoisonId;
    }
    for (float v : g.weights) {
      untouched &= std::isnan(v);
    }
    for (std::int32_t v : g.groups) {
      untouched &= v == kPoisonInt;
    }
    const std::string name = std::string(b.name) + " is refused";
    Report(name.c_str(),
           threw && untouched && hipGetLastError() == hipSuccess,
           untouched ? "no launch: ids, weights and table still poisoned"
                     : "a launch wrote into the buffers");
  }
}

/// Not a gate: the numbers steer the launcher.
void Timed(std::mt19937& rng) {
  for (const std::uint32_t rows : {1U, 16U}) {
    const Fixture f = RandomFixture(rows, kHidden, kExperts, 0.02, 0.5, rng());
    float* dx = Device(f.x);
    float* dr = Device(f.router);
    float* db = Device(f.bias);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(rows) * kUsed, 0);
    std::vector<float> weights(ids.size(), 0.0F);
    std::vector<std::int32_t> groups(k::GroupTableInts(rows * kUsed), 0);
    std::int32_t* dids = Device(ids);
    float* dw = Device(weights);
    std::int32_t* dg = Device(groups);
    k::MoeRouteArgs a{};
    a.x = dx;
    a.router = dr;
    a.bias = db;
    a.ids = dids;
    a.weights = dw;
    a.groups = dg;
    a.rows = rows;
    a.hidden = f.hidden;
    a.experts = f.experts;
    a.used = f.used;
    for (int i = 0; i < 20; ++i) {
      k::MoeRoute(a, nullptr);
    }
    hipEvent_t first = nullptr;
    hipEvent_t last = nullptr;
    HIP_CHECK(hipEventCreate(&first));
    HIP_CHECK(hipEventCreate(&last));
    HIP_CHECK(hipEventRecord(first));
    for (int i = 0; i < 200; ++i) {
      k::MoeRoute(a, nullptr);
    }
    HIP_CHECK(hipEventRecord(last));
    HIP_CHECK(hipEventSynchronize(last));
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, first, last));
    std::cout << "     route rows=" << rows << " " << Number(ms * 5.0)
              << " us per layer\n";
    HIP_CHECK(hipEventDestroy(first));
    HIP_CHECK(hipEventDestroy(last));
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dr),
                    static_cast<void*>(db), static_cast<void*>(dids),
                    static_cast<void*>(dw), static_cast<void*>(dg)}) {
      HIP_CHECK(hipFree(p));
    }
  }
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

  RandomRows(rng);
  BiasMovesSelectionOnly(rng);
  Ties(rng);
  SubEpsilonBoundary(rng);
  GroupTable(rng);
  Counts(rng);
  BatchInvariance(rng);
  SmallShapeAndOptionalOutputs(rng);
  Refusals();

  if (std::getenv("KOLIBRI_TIME") != nullptr) {
    Timed(rng);
  }

  if (failures != 0) {
    std::cout << failures << " routing checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 routing matches the reference over every case\n";
  return 0;
}
