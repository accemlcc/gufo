// Kolibri 1 attention on the gfx1151 matrix cores.
//
// Geometry of the released artifact: head_dim 128, 48 query heads over 4 key
// heads, so a key group is 12 query heads. Twelve is awkward for a 16-row
// fragment, sixteen query rows of one head are not, which gives two mappings:
//
//   rows-as-rows (prefill): a block covers four key heads' slice of the group
//     and two row groups of 16 queries, one wave each. Its 16x16 score tile is
//     16 queries by 16 keys, and a row's causal bound differs per row.
//   heads-as-rows (decode): with a single query row the twelve heads of one key
//     head become the fragment rows, so one wave holds one tile and all its rows
//     share a bound. The other seven waves of the block stage the next key tile,
//     which is where decode actually spends its time.
//
// Key ranges are long on purpose: the ten no-position layers read the whole
// history, up to 131072 keys. Those ranges are cut into splits that each produce
// a partial output plus its max and sum, and a merge kernel combines them, so a
// long context gets parallelism instead of one serial walker per key head.
//
// Per 16-key tile the block stages K row-major and V transposed in LDS once for
// every wave. Scores come out of the matrix cores in FP32 and pass through LDS
// for the online softmax: a fragment scatters one row's sixteen values over
// lanes, while max, sum and probabilities want them together. The probabilities
// return as the A operand of P times V.
//
// Fragment layout of wave32 v_wmma_f32_16x16x16_f16: an A lane holds sixteen
// contiguous reduction elements of row lane%16, a B lane holds sixteen
// contiguous reduction elements of column lane%16, and a C lane holds element i
// of row 2i + lane/16 at column lane%16. The reduction of Q times K transpose is
// the head dimension, so its sixteen accumulators add into one C fragment.
#include "src/models/kolibri1/kernels/rocm/attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gufo::models::kolibri1::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr std::uint32_t kKeys = 16;             // keys per tile
constexpr std::uint32_t kDim = 128;             // head_dim
constexpr std::uint32_t kSteps = kDim / kKeys;  // reduction steps of QK
constexpr std::uint32_t kWaves = 8;
constexpr std::uint32_t kThreads = 32 * kWaves;
constexpr std::uint32_t kGroup = 12;  // query heads per key head

constexpr std::uint32_t kKStride = kDim + 8;    // staged K rows, halves
constexpr std::uint32_t kVtStride = kKeys + 8;  // staged V columns, halves

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

struct Bounds {
  std::uint32_t lo;
  std::uint32_t hi;
};

/// Keys one query row may read: causal, capped by a truncated continuation, and
/// shortened to the window on a sliding layer. An unwindowed layer starts at
/// zero, which is what a layer without positional encoding needs.
__device__ __forceinline__ Bounds KeyBounds(const AttentionArgs& a,
                                            std::uint32_t position) {
  const std::uint32_t hi = min(position + 1, a.key_limit);
  std::uint32_t lo = 0;
  if (a.window != 0 && position + 1 > a.window) {
    lo = position + 1 - a.window;
  }
  return {min(lo, hi), hi};
}

/// Where the key range of this block starts and how wide it is. A split owns a
/// contiguous run of keys; the mask still decides the causal tail.
struct Split {
  std::uint32_t lo;
  std::uint32_t hi;
};

__device__ __forceinline__ Split BlockKeys(const AttentionArgs& a,
                                           std::uint32_t lo, std::uint32_t hi) {
  if (a.splits <= 1) {
    return {lo, hi};
  }
  const std::uint32_t span = hi - lo;
  const std::uint32_t chunk = (span + a.splits - 1) / a.splits;
  const std::uint32_t aligned = (chunk + kKeys - 1) & ~(kKeys - 1);
  // blockIdx.z names this block's slice, so one launch covers every split and
  // the merge pass reads the partial of the same index.
  const std::uint32_t from = lo + blockIdx.z * aligned;
  return {from, min(from + aligned, hi)};
}

template<bool kHeadsAsRows>
__global__ void __launch_bounds__(kThreads)
    __attribute__((amdgpu_waves_per_eu(2))) AttentionKernel(AttentionArgs a) {
  __shared__ __align__(16) __half k_lds[kKeys * kKStride];
  __shared__ __align__(16) __half vt_lds[kDim * kVtStride];
  __shared__ float s_lds[kWaves * 16][kKeys + 1];
  __shared__ __align__(16) __half p_lds[kWaves * 16][kKeys + 8];
  __shared__ float row_max[kWaves * 16];
  __shared__ float row_sum[kWaves * 16];
  __shared__ float row_scale[kWaves * 16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;  // this lane's fragment row

  // Prefill gives each wave one head and one group of sixteen query rows, so a
  // block covers two heads by four row groups - sixty-four query rows - and
  // grid.y runs over the head pairs of a key group. Decode has a single tile per
  // key head: wave 0 computes it while the other seven stage the key tiles,
  // which is where decode time actually goes.
  constexpr std::uint32_t kHeadsPerBlock = kHeadsAsRows ? 1 : 2;
  constexpr std::uint32_t kRowGroups = kWaves / kHeadsPerBlock;
  constexpr std::uint32_t kQueryRows = kRowGroups * 16;
  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t row0 =
      kHeadsAsRows ? 0 : query_start + (wave / kHeadsPerBlock) * 16;
  // The twelve heads of a key group form six pairs; blockIdx.y names one of
  // them, so a block never straddles two key heads.
  const std::uint32_t head =
      kHeadsAsRows
          ? blockIdx.y * kGroup
          : (blockIdx.y / (kGroup / kHeadsPerBlock)) * kGroup +
                (blockIdx.y % (kGroup / kHeadsPerBlock)) * kHeadsPerBlock +
                (wave % kHeadsPerBlock);
  const std::uint32_t live_rows =
      kHeadsAsRows ? kGroup : min(16U, row0 < a.rows ? a.rows - row0 : 0U);
  const bool computes = !kHeadsAsRows || wave == 0;
  const std::uint32_t kv_head = head / (a.heads / a.kv_heads);

  // Q narrowed to binary16: a lane carries its fragment row's sixteen dims of
  // one reduction step, so each load is one 32-byte span.
  v16h q_frag[kSteps];
#pragma unroll
  for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
    const std::uint32_t q_row = kHeadsAsRows ? query_start : row0 + sub;
    const std::uint32_t q_head = kHeadsAsRows ? head + sub : head;
    const bool live = computes && sub < live_rows && q_row < a.rows &&
                      q_head < a.heads;
    const float* src =
        a.q + (static_cast<std::size_t>(q_row) * a.heads + q_head) * kDim +
        ks * kKeys;
#pragma unroll
    for (std::uint32_t v = 0; v < kKeys; v += 4) {
      const float4 f = live ? *reinterpret_cast<const float4*>(src + v)
                            : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      q_frag[ks][v + 0] = static_cast<_Float16>(f.x);
      q_frag[ks][v + 1] = static_cast<_Float16>(f.y);
      q_frag[ks][v + 2] = static_cast<_Float16>(f.z);
      q_frag[ks][v + 3] = static_cast<_Float16>(f.w);
    }
  }

  // The key range every wave of this block walks. It must be the same for all
  // of them: the staged tile is shared memory and the barriers between staging,
  // scores and probabilities are block-wide, so a wave that counted a different
  // number of tiles would either read another wave's keys or miss a barrier.
  // Rows are consecutive positions, so the range spans the block's first row to
  // its last one that exists.
  const std::uint32_t block_first = query_start;
  const std::uint32_t block_last =
      kHeadsAsRows ? query_start : min(query_start + kQueryRows, a.rows) - 1;
  const Bounds first = KeyBounds(a, a.first_position + block_first);
  const Bounds last = KeyBounds(a, a.first_position + block_last);
  const Split keys = BlockKeys(a, first.lo & ~(kKeys - 1), last.hi);

  const std::size_t kv_stride = static_cast<std::size_t>(a.kv_heads) * kDim;
  const __half* k_cache = reinterpret_cast<const __half*>(a.k);
  const __half* v_cache = reinterpret_cast<const __half*>(a.v);
  const auto slot_of = [&](std::uint32_t key) {
    return a.ring != 0 ? key % a.ring : key;
  };

  if (tid < kWaves * 16) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.0F;
    row_scale[tid] = 1.0F;
  }
  v8f o_acc[kSteps] = {};

  // Every thread stages one 8-dim chunk of K and the same chunk of V: sixteen
  // bytes each, consecutive threads covering consecutive dims of consecutive
  // keys, so both loads are coalesced.
  const std::uint32_t chunk_key = tid / (kDim / 8);
  const std::uint32_t chunk_d8 = (tid % (kDim / 8)) * 8;

  __syncthreads();
  for (std::uint32_t key0 = keys.lo; key0 < keys.hi; key0 += kKeys) {
    const std::uint32_t position = key0 + chunk_key;
    uint4 k_chunk = make_uint4(0U, 0U, 0U, 0U);
    uint4 v_chunk = make_uint4(0U, 0U, 0U, 0U);
    if (position < keys.hi) {
      const std::size_t offset =
          slot_of(position) * kv_stride + static_cast<std::size_t>(kv_head) * kDim;
      k_chunk = *reinterpret_cast<const uint4*>(k_cache + offset + chunk_d8);
      v_chunk = *reinterpret_cast<const uint4*>(v_cache + offset + chunk_d8);
    }
    *reinterpret_cast<uint4*>(&k_lds[chunk_key * kKStride + chunk_d8]) = k_chunk;
    // V transposed: the eight dims of one key become eight rows of one column.
    const __half* vh = reinterpret_cast<const __half*>(&v_chunk);
#pragma unroll
    for (std::uint32_t d = 0; d < 8; ++d) {
      vt_lds[(chunk_d8 + d) * kVtStride + chunk_key] = vh[d];
    }
    __syncthreads();

    if (computes) {
      // Q times K transpose. The B operand wants column lane%16 with the
      // reduction contiguous, which is one staged key row; the head dimension
      // is the reduction, so the steps add into one accumulator.
      v8f score = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
        score = Wmma(q_frag[ks], LoadFrag(&k_lds[sub * kKStride + ks * kKeys]),
                     score);
      }
      // Publish the tile, scaled and masked, for the softmax pass. A C lane
      // holds element i of row 2i + lane/16 at column lane%16, so one lane
      // covers eight rows of the column its number names.
      const std::uint32_t key = key0 + sub;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t frag_row = 2 * i + (lane >> 4);
        const std::uint32_t query_row =
            kHeadsAsRows ? query_start : row0 + frag_row;
        const Bounds bounds = KeyBounds(a, a.first_position + query_row);
        const bool valid = frag_row < live_rows && query_row < a.rows &&
                           key >= bounds.lo && key < bounds.hi;
        s_lds[wave * 16 + frag_row][sub] = valid ? score[i] * a.scale : -INFINITY;
      }
    }
    __syncthreads();

    // Online softmax: one thread per fragment row over the tile's sixteen keys.
    // The probabilities are written for every tile, masked entries as zero: the
    // P times V product reads the whole tile and a tile left holding the
    // previous one's values would add keys that do not exist.
    if (tid < kWaves * 16) {
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t col = 0; col < kKeys; ++col) {
        tile_max = fmaxf(tile_max, s_lds[tid][col]);
      }
      const float carry = row_max[tid];
      const float best = fmaxf(carry, tile_max);
      const bool reachable = best != -INFINITY;
      const float base = reachable ? best : 0.0F;
      // A row that has not seen a key yet starts from nothing; a tile with no
      // visible key leaves the row exactly as it was.
      float rescale = 1.0F;
      if (reachable) {
        rescale = carry == -INFINITY ? 0.0F : expf(carry - best);
        row_max[tid] = best;
      }
      float sum = 0.0F;
#pragma unroll
      for (std::uint32_t col = 0; col < kKeys; ++col) {
        const float v = s_lds[tid][col];
        const float p = v == -INFINITY ? 0.0F : expf(v - base);
        p_lds[tid][col] = static_cast<_Float16>(p);
        sum += p;
      }
      row_scale[tid] = rescale;
      row_sum[tid] = row_sum[tid] * rescale + sum;
    }
    __syncthreads();

    if (computes) {
      // Rescale this wave's accumulators, then add P times V. The reduction of
      // that product is the key, so the same A fragment serves all eight output
      // blocks and B walks the transposed tile.
      const v16h p_frag = LoadFrag(&p_lds[wave * 16 + sub][0]);
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[ks][i] *= row_scale[wave * 16 + 2 * i + (lane >> 4)];
        }
        o_acc[ks] =
            Wmma(p_frag, LoadFrag(&vt_lds[(ks * kKeys + sub) * kVtStride]),
                 o_acc[ks]);
      }
    }
    __syncthreads();
  }

  // A split writes its partial output with the max and sum that go with it;
  // without splits the kernel normalizes and writes the result itself.
  if (a.splits > 1) {
    if (computes) {
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const std::uint32_t frag_row = 2 * i + (lane >> 4);
          const std::uint32_t query_row =
              kHeadsAsRows ? query_start : row0 + frag_row;
          const std::uint32_t out_head = kHeadsAsRows ? head + frag_row : head;
          if (frag_row >= live_rows || query_row >= a.rows) {
            continue;
          }
          const std::size_t base =
              ((static_cast<std::size_t>(blockIdx.z) * a.rows + query_row) *
                   a.heads +
               out_head) *
              (kDim + 2);
          a.partials[base + ks * kKeys + (lane & 15U)] = o_acc[ks][i];
          if (ks == 0 && (lane & 15U) == 0) {
            a.partials[base + kDim] = row_max[wave * 16 + frag_row];
            a.partials[base + kDim + 1] = row_sum[wave * 16 + frag_row];
          }
        }
      }
    }
    return;
  }

  if (computes) {
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSteps; ++ks) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t frag_row = 2 * i + (lane >> 4);
        const std::uint32_t query_row =
            kHeadsAsRows ? query_start : row0 + frag_row;
        const std::uint32_t out_head = kHeadsAsRows ? head + frag_row : head;
        if (frag_row >= live_rows || query_row >= a.rows) {
          continue;
        }
        const float total = row_sum[wave * 16 + frag_row];
        a.out[(static_cast<std::size_t>(query_row) * a.heads + out_head) * kDim +
              ks * kKeys + (lane & 15U)] =
            total > 0.0F ? o_acc[ks][i] / total : 0.0F;
      }
    }
  }
}

/// Combine the partial outputs of a split run. Each partial carries its own max
/// and sum, so the combination is the same rescaling the tile loop applies.
__global__ void __launch_bounds__(256)
    AttentionMergeKernel(AttentionArgs a) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t slots =
      static_cast<std::size_t>(a.rows) * a.heads * (kDim + 2);
  if (index >= slots) {
    return;
  }
  const std::size_t head_slot = index % (kDim + 2);
  const std::size_t row_head = index / (kDim + 2);
  // One partial per split, each holding the head dimension plus its max and sum.
  const std::size_t split_stride = slots;
  float best = -INFINITY;
  for (std::uint32_t s = 0; s < a.splits; ++s) {
    best = fmaxf(best,
                 a.partials[s * split_stride + row_head * (kDim + 2) + kDim]);
  }
  if (best == -INFINITY) {
    if (head_slot < kDim) {
      const std::uint32_t row = static_cast<std::uint32_t>(row_head / a.heads);
      const std::uint32_t head = static_cast<std::uint32_t>(row_head % a.heads);
      a.out[(static_cast<std::size_t>(row) * a.heads + head) * kDim + head_slot] =
          0.0F;
    }
    return;
  }
  float accumulator = 0.0F;
  float total = 0.0F;
  for (std::uint32_t s = 0; s < a.splits; ++s) {
    const float* partial =
        a.partials + s * split_stride + row_head * (kDim + 2);
    const float scale = expf(partial[kDim] - best);
    if (head_slot < kDim) {
      accumulator += partial[head_slot] * scale;
    }
    total += partial[kDim + 1] * scale;
  }
  if (head_slot < kDim) {
    const std::uint32_t row = static_cast<std::uint32_t>(row_head / a.heads);
    const std::uint32_t head = static_cast<std::uint32_t>(row_head % a.heads);
    a.out[(static_cast<std::size_t>(row) * a.heads + head) * kDim + head_slot] =
        total > 0.0F ? accumulator / total : 0.0F;
  }
}

}  // namespace

constexpr std::uint32_t kMaxSplits = 16;
/// A split shorter than this is not worth its share of the merge.
constexpr std::uint32_t kKeysPerSplit = 64;

std::uint32_t PickSplits(std::uint32_t keys) noexcept {
  // Four key heads alone cannot fill the GPU on a long context, so the range is
  // cut until every compute unit has a block - but never so finely that a split
  // walks fewer than four key tiles.
  const std::uint32_t by_span = keys / kKeysPerSplit;
  return std::clamp(by_span == 0 ? 1U : by_span, 1U, kMaxSplits);
}

std::uint32_t MaxSplits(std::uint32_t capacity) noexcept {
  return PickSplits(capacity);
}

void Attention(const AttentionArgs& args, hipStream_t stream) {
  if (args.head_dim != kDim) {
    throw std::invalid_argument(
        "kolibri1 attention is written for head_dim 128");
  }
  if (args.kv_heads == 0 || args.heads % args.kv_heads != 0) {
    throw std::invalid_argument(
        "kolibri1 attention needs query heads grouped by the key heads");
  }
  if (args.heads / args.kv_heads != kGroup) {
    throw std::invalid_argument(
        "kolibri1 attention expects twelve query heads per key head");
  }
  if (args.rows == 0) {
    return;
  }
  if (args.rows == 1) {
    // Decode: one key head per block row, its twelve heads as fragment rows, and
    // as many slices along the keys as the visible range supports.
    AttentionArgs split = args;
    const std::uint32_t limit =
        std::min(args.key_limit, args.first_position + args.rows);
    const std::uint32_t visible =
        args.window != 0 ? std::min(args.window, limit) : limit;
    split.splits = PickSplits(visible);
    const dim3 grid(1, args.kv_heads, split.splits);
    AttentionKernel<true><<<grid, kThreads, 0, stream>>>(split);
    if (split.splits > 1) {
      // One thread per output element plus the max and sum of each row and head;
      // each thread walks the splits itself.
      const std::size_t slots =
          static_cast<std::size_t>(args.rows) * args.heads * (kDim + 2);
      AttentionMergeKernel<<<static_cast<std::uint32_t>((slots + 255) / 256),
                             256, 0, stream>>>(split);
    }
    return;
  }
  AttentionArgs single = args;
  single.splits = 1;
  const std::uint32_t row_groups =
      (args.rows + AttentionQueryRowsPerBlock() - 1) /
      AttentionQueryRowsPerBlock();
  const dim3 grid(row_groups, args.kv_heads * (kGroup / 2));
  AttentionKernel<false><<<grid, kThreads, 0, stream>>>(single);
}

std::size_t AttentionPartialFloats(std::uint32_t rows, std::uint32_t heads,
                                   std::uint32_t head_dim,
                                   std::uint32_t capacity) noexcept {
  // The largest split count the launcher may choose, so the caller allocates
  // once for the deepest context it intends to run.
  return static_cast<std::size_t>(MaxSplits(capacity)) * rows * heads *
         (head_dim + 2);
}

}  // namespace gufo::models::kolibri1::rocm
