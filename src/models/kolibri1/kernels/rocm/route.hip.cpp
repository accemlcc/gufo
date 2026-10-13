// Kolibri 1's router: the float matrix product over the layer input, the six
// experts per row chosen by `logit + bias`, and the weight each of them
// contributes, which is the sigmoid of its own unbiased logit.
//
// Two launches, because the choice needs every logit of its row while the
// logits are produced by blocks that split the experts between them: the first
// computes and publishes them, the second reads them back and selects. The
// published buffer is the only copy, so the selection compares exactly the
// numbers a verifier reads, and a logit that lands between two representable
// scores is a tie for both.
//
// One row of the router is 2560 products, and a single float chain that long
// moves a logit by a few times 1e-7 — wider than the gap between the sixth and
// the seventh expert of a typical row, so the batch a row happened to run in
// would decide its experts. The accumulation is split instead: four float
// chains per lane, one per slot of a float4, so no chain is longer than
// hidden / 128 additions, and the cross-lane sum in double, where the terms
// have already grown to the size of the answer. Carrying the products in double
// all the way is exact and cost 261 us for a full sixteen rows against the 105
// us this takes; the fp64 adders of this generation trail the memory that feeds
// them, and the router is otherwise 384 * 2560 * 4 = 3.9 MB of weights a layer.
#include "src/models/kolibri1/kernels/rocm/moe.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "src/core/hip/hip_utils.hpp"

namespace gufo::models::kolibri1::rocm {
namespace {

constexpr std::uint32_t kWave = 32;

__device__ __forceinline__ double WaveSum(double v) {
#pragma unroll
  for (std::uint32_t m = kWave / 2; m > 0; m >>= 1) {
    v += __shfl_xor(v, static_cast<int>(m), kWave);
  }
  return v;
}

/// Sigmoid of the logit as stored, in double, narrowed once at the end. The
/// bias is not an argument here on purpose: it moves the choice and nothing
/// else.
__device__ __forceinline__ float Sigmoid(float logit) {
  return static_cast<float>(1.0 / (1.0 + exp(-static_cast<double>(logit))));
}

// ---------------------------------------------------------------------------
// Router logits
// ---------------------------------------------------------------------------

constexpr std::uint32_t kRouterWaves = 4;
constexpr std::uint32_t kRouterThreads = kWave * kRouterWaves;

/// One wave per expert: lane l reads the float4s of router[e] at l + 32 i, so
/// a wavefront touches 512 contiguous bytes, and dots them with the same slice
/// of every row of x.
///
/// The row of the router is not held in registers. Twenty float4 per lane
/// would keep it out of the caches for a batch of rows, at eighty registers per
/// lane, which is what limits the waves in flight against the memory latency
/// the same kernel depends on; the re-read of a 10 KB row lands in the CU's own
/// cache instead and costs no bandwidth a row of x does not cost anyway.
__global__ void __launch_bounds__(kRouterThreads)
RouterKernel(const float* __restrict__ x, const float* __restrict__ router,
             float* __restrict__ logits, std::uint32_t rows,
             std::uint32_t hidden, std::uint32_t experts) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t e = blockIdx.x * kRouterWaves + threadIdx.x / kWave;
  // A wave past the last expert leaves rather than waits: this kernel has no
  // barrier, so idling is the only thing it could do with its time.
  if (e >= experts) {
    return;
  }
  const std::uint32_t quads = hidden / 4;  // a multiple of the wave
  const float4* const wr =
      reinterpret_cast<const float4*>(router +
                                      static_cast<std::size_t>(e) * hidden);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const float4* const xr =
        reinterpret_cast<const float4*>(x +
                                        static_cast<std::size_t>(r) * hidden);
    float4 part = {0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint32_t i = lane; i < quads; i += kWave) {
      const float4 w = wr[i];
      const float4 v = xr[i];
      part.x = __builtin_fmaf(w.x, v.x, part.x);
      part.y = __builtin_fmaf(w.y, v.y, part.y);
      part.z = __builtin_fmaf(w.z, v.z, part.z);
      part.w = __builtin_fmaf(w.w, v.w, part.w);
    }
    double acc =
        ((static_cast<double>(part.x) + static_cast<double>(part.y)) +
         (static_cast<double>(part.z) + static_cast<double>(part.w)));
    acc = WaveSum(acc);
    if (lane == 0) {
      logits[static_cast<std::size_t>(r) * experts + e] =
          static_cast<float>(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// Selection, weights, counts and the group table
// ---------------------------------------------------------------------------

/// One block covers kSelectRows rows: sixteen waves, one per row. A launch may
/// span more of the forward than that - the rows are independent here - but
/// then the table cannot be built inside the blocks that made it, and goes to
/// GatherGroupsKernel instead.
constexpr std::uint32_t kSelectThreads = 512;
static_assert(kSelectThreads / kWave == kSelectRows,
              "the selection block must hold exactly its rows in waves");
/// What a lane holds of the best it has seen. The list is a fixed depth so the
/// insert and the pop stay unrolled at compile time: a depth that grew with
/// `used` would be indexed at run time and spill.
constexpr std::uint32_t kMaxUsed = 16;
constexpr std::int32_t kNoExpert = 0x7FFFFFFF;

/// Larger score wins, and an equal score keeps the lower expert index. Every
/// comparison of the selection goes through this, so the tie rule cannot differ
/// between filling the lists and reducing them.
__device__ __forceinline__ bool Better(double a, std::int32_t ia, double b,
                                       std::int32_t ib) {
  return a > b || (a == b && ia < ib);
}

/// One wave per row.
///
/// The ported layout from gemma 4 holds `experts / 32` logits per lane, which
/// is twelve here: the registers then grow with the artifact and every round
/// re-scans all of them, so the work goes as experts * used instead of experts.
/// Instead each wave scans its row's logits once in stride-32 runs and keeps a
/// running best of kMaxUsed per lane -- sixteen candidates, which is more than
/// any launch may select, so the union of the lanes' lists always holds the
/// row's whole answer -- and then `used` rounds of a wave reduction pop the
/// experts in order. A row's numbers therefore depend on its own logits only.
__global__ void __launch_bounds__(kSelectThreads)
SelectKernel(const float* __restrict__ logits, const float* __restrict__ bias,
             std::int32_t* __restrict__ ids, float* __restrict__ weights,
             std::uint32_t* __restrict__ counts,
             std::int32_t* __restrict__ groups, std::uint32_t rows,
             std::uint32_t experts, std::uint32_t used) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t wave = threadIdx.x / kWave;
  // The row this wave selects out of the whole launch, so a grid of blocks
  // walks the forward instead of only its first kSelectRows.
  const std::uint32_t r = blockIdx.x * kSelectRows + wave;

  double best_score[kMaxUsed];
  std::int32_t best_id[kMaxUsed];
#pragma unroll
  for (std::uint32_t j = 0; j < kMaxUsed; ++j) {
    best_score[j] = -INFINITY;
    best_id[j] = kNoExpert;
  }

  const float* row = nullptr;
  if (r < rows) {
    row = logits + static_cast<std::size_t>(r) * experts;
    for (std::uint32_t e = lane; e < experts; e += kWave) {
      // The sum is exact in double: two floats added there lose nothing, so a
      // tie here is a tie in the arithmetic and not an artefact of rounding.
      const double score = static_cast<double>(row[e]) +
                           static_cast<double>(bias[e]);
      const std::int32_t id = static_cast<std::int32_t>(e);
      if (!Better(score, id, best_score[kMaxUsed - 1],
                  best_id[kMaxUsed - 1])) {
        continue;  // the common case: worse than this lane's sixteenth best
      }
      // Sink it through the list by successive exchange from the front down:
      // every exchange hands the entry below it to the next slot, so the list
      // stays sorted and the seventeenth candidate leaves it at the end. Going
      // the other way round would exchange once and stop, because the carry
      // turns worse on its very first swap.
      double carry_score = score;
      std::int32_t carry_id = id;
#pragma unroll
      for (std::uint32_t j = 0; j < kMaxUsed; ++j) {
        if (Better(carry_score, carry_id, best_score[j], best_id[j])) {
          const double held = best_score[j];
          const std::int32_t held_id = best_id[j];
          best_score[j] = carry_score;
          best_id[j] = carry_id;
          carry_score = held;
          carry_id = held_id;
        }
      }
    }

    for (std::uint32_t j = 0; j < used; ++j) {
      double score = best_score[0];
      std::int32_t id = best_id[0];
#pragma unroll
      for (std::uint32_t m = kWave / 2; m > 0; m >>= 1) {
        const double other_score = __shfl_xor(score, static_cast<int>(m), kWave);
        const std::int32_t other_id = __shfl_xor(id, static_cast<int>(m), kWave);
        if (Better(other_score, other_id, score, id)) {
          score = other_score;
          id = other_id;
        }
      }
      // Exactly one lane holds the winner and only it gives it up. The lists of
      // different lanes are disjoint, so no two of them can agree on an id.
      if (best_id[0] == id) {
#pragma unroll
        for (std::uint32_t q = 0; q + 1 < kMaxUsed; ++q) {
          best_score[q] = best_score[q + 1];
          best_id[q] = best_id[q + 1];
        }
        best_score[kMaxUsed - 1] = -INFINITY;
        best_id[kMaxUsed - 1] = kNoExpert;
      }
      if (lane == j) {  // used <= kMaxUsed <= 16, so one lane per round
        const std::size_t slot = static_cast<std::size_t>(r) * used + j;
        ids[slot] = id;
        weights[slot] = Sigmoid(row[id]);
        if (counts != nullptr) {
          atomicAdd(counts + id, 1U);
        }
      }
    }
  }

  if (groups == nullptr) {
    return;  // uniform across the block: no barrier is left unmet
  }

  // The table wants the slots of an expert together and in increasing order,
  // while they were written row by row. One thread per expert collects its own
  // out of shared memory, and a block scan over the experts that have any
  // places the groups, which puts them in increasing expert order by
  // construction. Reached only for a launch of at most kSelectRows rows, which
  // is what makes this block's ids the whole table's ids; beyond it the caller
  // passes no table and the gather builds it from the counts.
  __shared__ std::int32_t s_ids[kSelectRows * kMaxUsed];
  __shared__ int s_scan[kSelectThreads];
  __syncthreads();  // every wave's ids are visible to this block
  const std::uint32_t slots = rows * used;
  for (std::uint32_t s = threadIdx.x; s < slots; s += kSelectThreads) {
    s_ids[s] = ids[s];
  }
  __syncthreads();

  const std::uint32_t chunk =
      (experts + kSelectThreads - 1) / kSelectThreads;
  const std::uint32_t lo = threadIdx.x * chunk;
  const std::uint32_t hi = lo + chunk < experts ? lo + chunk : experts;
  const auto count_of = [&](std::int32_t e) {
    int n = 0;
    for (std::uint32_t s = 0; s < slots; ++s) {
      n += s_ids[s] == e ? 1 : 0;
    }
    return n;
  };

  int mine = 0;
  for (std::uint32_t e = lo; e < hi; ++e) {
    mine += count_of(static_cast<std::int32_t>(e)) > 0 ? 1 : 0;
  }
  s_scan[threadIdx.x] = mine;
  __syncthreads();
  for (std::uint32_t off = 1; off < kSelectThreads; off <<= 1) {
    const int behind =
        threadIdx.x >= off ? s_scan[threadIdx.x - off] : 0;
    __syncthreads();
    s_scan[threadIdx.x] += behind;
    __syncthreads();
  }
  int at = s_scan[threadIdx.x] - mine;
  for (std::uint32_t e = lo; e < hi; ++e) {
    const std::int32_t expert = static_cast<std::int32_t>(e);
    const int n = count_of(expert);
    if (n == 0) {
      continue;  // an expert with no slot is not mentioned by the table at all
    }
    // At most one slot of a group per row, so a group never needs more than
    // the kMaxGroupSlots the kGroupInts stride leaves room for.
    std::int32_t* g = groups + 1 + static_cast<std::size_t>(at) * kGroupInts;
    g[0] = expert;
    g[1] = n;
    int w = 0;
    for (std::uint32_t s = 0; s < slots; ++s) {
      if (s_ids[s] == expert) {
        g[2 + w++] = static_cast<std::int32_t>(s);
      }
    }
    ++at;
  }
  if (threadIdx.x == kSelectThreads - 1) {
    groups[0] = s_scan[kSelectThreads - 1];
  }
}

/// One block gathers the group table of a selection that ran over more rows
/// than a single block holds.
///
/// Thread e owns expert e. Its slot count is the count the selection
/// accumulated - the selection is the only writer and it counted every id it
/// published - and its slots are those ids in increasing order, which is what a
/// forward scan of the slots yields. The placement of the groups is a block
/// scan over the experts that hold any, so the table lists experts in
/// increasing order and never mentions an empty one, exactly as the in-block
/// build of a narrower launch does; the grouped projection cannot tell which of
/// the two wrote what it reads.
constexpr std::uint32_t kGatherThreads = kGatherExperts;
__global__ void __launch_bounds__(kGatherThreads)
    GatherGroupsKernel(const std::uint32_t* __restrict__ counts,
                       const std::int32_t* __restrict__ ids,
                       std::int32_t* __restrict__ groups, std::uint32_t experts,
                       std::uint32_t slots) {
  __shared__ std::int32_t s_ids[kMaxGroupSlots * kMaxUsed];
  __shared__ int s_scan[kGatherThreads];
  for (std::uint32_t s = threadIdx.x; s < slots; s += kGatherThreads) {
    s_ids[s] = ids[s];
  }
  const std::uint32_t e = threadIdx.x;
  const int mine = e < experts && counts[e] > 0 ? 1 : 0;
  s_scan[threadIdx.x] = mine;
  __syncthreads();
  for (std::uint32_t off = 1; off < kGatherThreads; off <<= 1) {
    const int behind = threadIdx.x >= off ? s_scan[threadIdx.x - off] : 0;
    __syncthreads();
    s_scan[threadIdx.x] += behind;
    __syncthreads();
  }
  if (mine == 0) {
    if (e == kGatherThreads - 1) {
      groups[0] = s_scan[kGatherThreads - 1];
    }
    return;  // an expert with no slot is not mentioned by the table at all
  }
  std::int32_t* g =
      groups + 1 + static_cast<std::size_t>(s_scan[e] - 1) * kGroupInts;
  g[0] = static_cast<std::int32_t>(e);
  const std::uint32_t n = counts[e];
  g[1] = static_cast<std::int32_t>(n);
  int w = 0;
  for (std::uint32_t s = 0; s < slots; ++s) {
    if (s_ids[s] == static_cast<std::int32_t>(e)) {
      // At most one slot of a group per row, so a group never needs more than
      // the kMaxGroupSlots the kGroupInts stride leaves room for.
      g[2 + w++] = static_cast<std::int32_t>(s);
    }
  }
  if (e == kGatherThreads - 1) {
    groups[0] = s_scan[kGatherThreads - 1];
  }
}

}  // namespace

void MoeRoute(const MoeRouteArgs& args, hipStream_t stream) {
  if (args.rows == 0) {
    return;
  }
  if (args.rows > kMaxGroupSlots) {
    throw std::invalid_argument(
        "MoeRoute covers at most " + std::to_string(kMaxGroupSlots) + " rows");
  }
  // A table over more rows than one block selects cannot be built from a
  // block's own ids: the gather reads the counts, which the caller then has to
  // provide, and it gives each expert one thread, which bounds the experts of
  // that path alone.
  if (args.groups != nullptr && args.rows > kSelectRows) {
    if (args.counts == nullptr) {
      throw std::invalid_argument(
          "MoeRoute gathers a wide group table from the counts");
    }
    if (args.experts > kGatherExperts) {
      throw std::invalid_argument(
          "MoeRoute's gather gives one thread to each expert, so a wide table "
          "covers at most " +
              std::to_string(kGatherExperts) + " experts");
    }
  }
  if (args.used > args.experts) {
    throw std::invalid_argument(
        "MoeRoute cannot use more experts than the model has");
  }
  if (args.used == 0 || args.used > kMaxUsed) {
    throw std::invalid_argument(
        "MoeRoute selects between one and sixteen experts per row");
  }
  if (args.hidden == 0 || args.hidden % (4 * kWave) != 0) {
    throw std::invalid_argument(
        "MoeRoute needs a hidden size that is a multiple of 128");
  }
  if (args.x == nullptr || args.router == nullptr || args.bias == nullptr ||
      args.ids == nullptr || args.weights == nullptr) {
    throw std::invalid_argument(
        "MoeRoute needs the input, the router, the bias, ids and weights");
  }

  // The caller's own buffer serves as the hand-over when it asked for the raw
  // scores; otherwise the scratch is stream-ordered, so two requests on two
  // streams never share one table of logits.
  float* logits = args.logits;
  float* scratch = nullptr;
  if (logits == nullptr) {
    HIP_CHECK(hipMallocAsync(
        &scratch,
        static_cast<std::size_t>(args.rows) * args.experts * sizeof(float),
        stream));
    logits = scratch;
  }
  const dim3 router_grid((args.experts + kRouterWaves - 1) / kRouterWaves);
  RouterKernel<<<router_grid, kRouterThreads, 0, stream>>>(
      args.x, args.router, logits, args.rows, args.hidden, args.experts);
  // One block of rows each; the table of a single block is built inside it,
  // and the one of several blocks by the gather over the published ids.
  const dim3 select_grid((args.rows + kSelectRows - 1) / kSelectRows);
  const bool in_block_table = args.rows <= kSelectRows;
  SelectKernel<<<select_grid, kSelectThreads, 0, stream>>>(
      logits, args.bias, args.ids, args.weights, args.counts,
      in_block_table ? args.groups : nullptr, args.rows, args.experts,
      args.used);
  if (args.groups != nullptr && !in_block_table) {
    GatherGroupsKernel<<<1, kGatherThreads, 0, stream>>>(
        args.counts, args.ids, args.groups, args.experts,
        args.rows * args.used);
  }
  if (scratch != nullptr) {
    HIP_CHECK(hipFreeAsync(scratch, stream));
  }
}

}  // namespace gufo::models::kolibri1::rocm
