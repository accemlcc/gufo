// Kolibri 1's grouped projection: the routed experts and the ordinary
// projections in one kernel over the two super-block formats the artifact
// carries, Q4_K and Q6_K.
//
// A workgroup covers kRowsPerBlock rows of one expert's matrix and one group of
// the table, so an expert's weights are read once per pass of slots. The
// mapping inside the block: four lanes take the four 64-value quarters of a
// 256-value super-block and together hold one output row, eight rows per wave,
// and where the width wants it the kSplit waves of a row group take interleaved
// super-blocks of the reduction and are summed in wave order.
//
// What makes this usable for a mixture model is that a slot's arithmetic is
// fixed by its own row, the shape and the format:
//
//   * the unit walk is `unit = slice; unit += kSplit`, chosen from k alone,
//   * the four quarters join with a butterfly, so all four lanes of a quad get
//     the same bits and the pairing order never depends on the neighbours,
//   * the split slices are added in increasing order,
//   * a slot's accumulator starts at zero inside the pass that carries it, so
//     the pass width only decides how often the weights are re-read.
//
// Nothing in that list reads the table beyond the slot's own index, which is
// why a token comes out bit-identical whether it shares the launch with one
// group or with fifteen: a batch boundary may regroup, not round. Hence this
// file is compiled with -fno-fast-math -ffp-contract=off (CMakeLists.txt): the
// FMA order is the result.
//
// The Gemma kernel this is cut from carries a second specialization for groups
// that hold one slot, the shape decode runs. It is not ported: with the chain
// fixed by the shape a narrower pass is the same addition with fewer
// accumulators, so the specialization would only double what has to stay
// correct, and the check asserts a lone slot equals itself in a sixteen-slot
// table at every width the launcher picks.
//
// The alignment rules the decoders depend on are stated where they are used: a
// Q4_K block is 144 bytes and a Q4_K row 144 * k / 256, both multiples of 16,
// sixteen-byte loads of a row are aligned whenever the tensor is; a Q6_K block
// is 210 bytes, so every other block of a row starts two bytes past a word and
// no window of it is aligned. Those windows are read from the enclosing words
// with the byte-alignment instruction, and the launcher only accepts a width
// whose row size is a multiple of four so that the round-down never leaves the
// block.
#include "src/models/kolibri1/kernels/rocm/moe.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace gufo::models::kolibri1::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;
/// Waves a row's super-blocks are shared between when there are enough of them.
constexpr int kSplitWaves = 4;

// ---------------------------------------------------------------------------
// One lane's weights
// ---------------------------------------------------------------------------

__device__ __forceinline__ float Half(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  const auto h = static_cast<unsigned short>(bits & 0xFFFFU);
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

/// ggml's scale and minimum of 32-value sub-block j, read out of the three
/// header words that hold the twelve packed bytes instead of byte by byte.
__device__ __forceinline__ void ScaleMin(std::uint32_t w0, std::uint32_t w1,
                                         std::uint32_t w2, int j, float* sc,
                                         float* m) {
  const std::uint32_t shift = 8U * static_cast<std::uint32_t>(j & 3);
  const std::uint32_t a = (w0 >> shift) & 0xFFU;
  const std::uint32_t b = (w1 >> shift) & 0xFFU;
  const std::uint32_t c = (w2 >> shift) & 0xFFU;
  const bool low = j < 4;
  *sc = static_cast<float>(low ? (a & 63U) : ((c & 0xFU) | ((a >> 6) << 4)));
  *m = static_cast<float>(low ? (b & 63U) : ((c >> 4) | ((b >> 6) << 4)));
}

/// Sixteen payload bytes at an address of unknown alignment: the enclosing
/// words are read and the window taken out with the byte-alignment instruction,
/// which selects four bytes out of the pair with the lower address in its
/// second operand. Every window a Q6_K decode asks for starts at a multiple of
/// 16 inside the block, so with at most three trailing bytes the read stops at
/// 196 of the block's 210 and never leaves it.
__device__ __forceinline__ void Load16(const std::uint8_t* p,
                                       std::uint32_t (&out)[4]) {
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
  const int shift = static_cast<int>(addr & 3U);
  const auto* words = reinterpret_cast<const std::uint32_t*>(p - shift);
  const std::uint32_t w[5] = {words[0], words[1], words[2], words[3], words[4]};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = __builtin_amdgcn_alignbyte(w[i + 1], w[i], shift);
  }
}

/// out[i] = scale * v + offset with the nibble of the code at `shift`: the
/// sub-block's scale is folded into one product, and the minimum of Q4_K enters
/// as the offset the multiply is rounded against.
__device__ __forceinline__ void Dequant16(const std::uint8_t* q, int shift,
                                          float scale, float offset,
                                          float* out) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t v = (q[i] >> shift) & 0xFU;
    out[i] = __builtin_fmaf(scale, static_cast<float>(v), offset);
  }
}

/// The same for a six-bit code, whose top two bits sit in another byte.
__device__ __forceinline__ void Dequant16Six(const std::uint8_t* q, int shift,
                                             const std::uint8_t* high,
                                             int hshift, float scale,
                                             float offset, float* out) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t v =
        ((q[i] >> shift) & 0xFU) | (((high[i] >> hshift) & 3U) << 4);
    out[i] = __builtin_fmaf(scale, static_cast<float>(v), offset);
  }
}

/// acc += sum_i w[i] * x[i] as one FMA chain over n values, four per load. The
/// order is the chain itself: no reassociation, no vector reduction.
template <int N>
__device__ __forceinline__ float Accumulate(float acc, const float* w,
                                            const float* x) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    acc = __builtin_fmaf(w[i], v.x, acc);
    acc = __builtin_fmaf(w[i + 1], v.y, acc);
    acc = __builtin_fmaf(w[i + 2], v.z, acc);
    acc = __builtin_fmaf(w[i + 3], v.w, acc);
  }
  return acc;
}

/// Storage of each format: bytes per 256-value super-block, the lanes a row is
/// cut into and the values a lane decodes.
template <WeightFormat F>
struct Traits;

template <>
struct Traits<WeightFormat::kQ4_K> {
  static constexpr int kBytes = 144;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
};

template <>
struct Traits<WeightFormat::kQ6_K> {
  static constexpr int kBytes = 210;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
};

template <WeightFormat F>
__host__ __device__ constexpr std::size_t RowBytes(std::uint32_t k) {
  return static_cast<std::size_t>(k) / 256 * Traits<F>::kBytes;
}

/// Q4_K sub-block pair p of a super-block, one header load for 64 values: the
/// low nibbles of qs bytes [32 p, +32) are sub-block 2 p and the high nibbles
/// sub-block 2 p + 1, so w[i] belongs to activation 64 p + i.
__device__ __forceinline__ void DecodeQ4K(const std::uint8_t* block, int p,
                                          float (&w)[64]) {
  const uint4 header = *reinterpret_cast<const uint4*>(block);
  const float d = HalfBits(header.x);
  const float dmin = HalfBits(header.x >> 16);
  float sc0, m0, sc1, m1;
  ScaleMin(header.y, header.z, header.w, 2 * p, &sc0, &m0);
  ScaleMin(header.y, header.z, header.w, 2 * p + 1, &sc1, &m1);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint4 raw_q =
        *reinterpret_cast<const uint4*>(block + 16 + 32 * p + 16 * h);
    const auto* q = reinterpret_cast<const std::uint8_t*>(&raw_q);
    Dequant16(q, 0, d * sc0, -(dmin * m0), w + 16 * h);
    Dequant16(q, 4, d * sc1, -(dmin * m1), w + 32 + 16 * h);
  }
}

/// Q6_K task t = 2 n + h: ql bytes [64 n + 16 h, +16) and
/// [64 n + 32 + 16 h, +16) with qh bytes [128 + 32 n + 16 h, +16), which are
/// the values 128 n + 16 h + 32 r + i under scale scales[8 n + h + 2 r] and the
/// offset 32. The four quarters of a super-block tile its payload exactly, so a
/// launch reads every weight byte once.
__device__ __forceinline__ void DecodeQ6K(const std::uint8_t* block, int t,
                                          float (&w)[64]) {
  const int n = t >> 1;
  const int h = t & 1;
  const float d = Half(block + 208);
  const auto* sc =
      reinterpret_cast<const std::int8_t*>(block + 192) + 8 * n + h;
  std::uint32_t raw_a[4], raw_b[4], raw_h[4];
  Load16(block + 64 * n + 16 * h, raw_a);
  Load16(block + 64 * n + 32 + 16 * h, raw_b);
  Load16(block + 128 + 32 * n + 16 * h, raw_h);
  const auto* ql_a = reinterpret_cast<const std::uint8_t*>(raw_a);
  const auto* ql_b = reinterpret_cast<const std::uint8_t*>(raw_b);
  const auto* qh = reinterpret_cast<const std::uint8_t*>(raw_h);
  const float s0 = d * static_cast<float>(sc[0]);
  const float s1 = d * static_cast<float>(sc[2]);
  const float s2 = d * static_cast<float>(sc[4]);
  const float s3 = d * static_cast<float>(sc[6]);
  // (q - 32) * s as s * q - 32 s, with 32 s exact.
  Dequant16Six(ql_a, 0, qh, 0, s0, -32.0F * s0, w);
  Dequant16Six(ql_b, 0, qh, 2, s1, -32.0F * s1, w + 16);
  Dequant16Six(ql_a, 4, qh, 4, s2, -32.0F * s2, w + 32);
  Dequant16Six(ql_b, 4, qh, 6, s3, -32.0F * s3, w + 48);
}

/// Lane `t`'s decoded weights of super-block `unit` of a row and the activation
/// index of w[0]. The launcher refused every k the blocks do not divide.
template <WeightFormat F>
__device__ __forceinline__ void Decode(const std::uint8_t* row,
                                       std::uint32_t unit, int t,
                                       float (&w)[Traits<F>::kValues],
                                       std::uint32_t* offset) {
  const std::uint8_t* block =
      row + static_cast<std::size_t>(unit) * Traits<F>::kBytes;
  if constexpr (F == WeightFormat::kQ6_K) {
    DecodeQ6K(block, t, w);
    *offset = unit * 256 + 128 * static_cast<std::uint32_t>(t >> 1) +
              16 * static_cast<std::uint32_t>(t & 1);
  } else {
    DecodeQ4K(block, t, w);
    *offset = unit * 256 + 64 * static_cast<std::uint32_t>(t);
  }
}

/// acc += w . x for one decoded lane: runs of 16 values, 32 apart for Q6_K
/// because that is the order its super-block stores them in.
template <WeightFormat F>
__device__ __forceinline__ float Contract(float acc,
                                          const float (&w)[Traits<F>::kValues],
                                          const float* x) {
  constexpr int kRuns = Traits<F>::kValues / 16;
  constexpr int kStride = F == WeightFormat::kQ6_K ? 32 : 16;
#pragma unroll
  for (int r = 0; r < kRuns; ++r) {
    acc = Accumulate<16>(acc, w + 16 * r, x + kStride * r);
  }
  return acc;
}

/// The same addresses as Contract, storing instead: one dequantized lane of the
/// table written where its values belong in a row.
template <WeightFormat F>
__device__ __forceinline__ void Scatter(float* row,
                                        const float (&w)[Traits<F>::kValues],
                                        std::uint32_t offset) {
  constexpr int kRuns = Traits<F>::kValues / 16;
  constexpr int kStride = F == WeightFormat::kQ6_K ? 32 : 16;
#pragma unroll
  for (int r = 0; r < kRuns; ++r) {
    auto* out = reinterpret_cast<float4*>(row + offset + kStride * r);
    out[0] = make_float4(w[16 * r], w[16 * r + 1], w[16 * r + 2],
                         w[16 * r + 3]);
    out[1] = make_float4(w[16 * r + 4], w[16 * r + 5], w[16 * r + 6],
                         w[16 * r + 7]);
    out[2] = make_float4(w[16 * r + 8], w[16 * r + 9], w[16 * r + 10],
                         w[16 * r + 11]);
    out[3] = make_float4(w[16 * r + 12], w[16 * r + 13], w[16 * r + 14],
                         w[16 * r + 15]);
  }
}

/// Sum of the lanes one output row is cut into. The butterfly stays inside that
/// group of lanes, and IEEE addition being commutative, every one of them ends
/// with the same bits: which lane stores the row does not change the value.
template <int kTasks>
__device__ __forceinline__ float LaneSum(float acc) {
#pragma unroll
  for (int offset = kTasks / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Grouped projection
// ---------------------------------------------------------------------------

/// A workgroup: kWaves / kSplit row groups of kRowsPerWave rows, the kSplit
/// waves of a row group taking interleaved super-blocks. blockIdx.y is the
/// group of the table, and a group listing no slots writes nothing at all,
/// which is how an expert no row of this batch chose leaves the launch.
template <WeightFormat F, int kSplit, int kSlots>
__global__ void __launch_bounds__(kThreads) GroupedGemvKernel(
    const std::uint8_t* __restrict__ w, const std::int32_t* __restrict__ groups,
    const float* __restrict__ x, float* __restrict__ y,
    std::uint32_t slots_per_row, std::uint32_t m, std::uint32_t k) {
  using T = Traits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  constexpr int kRowGroups = kWaves / kSplit;
  constexpr int kRowsPerBlock = kRowsPerWave * kRowGroups;
  __shared__ float partial[kSlots][kSplit][kRowGroups][kWave];

  if (blockIdx.y >= static_cast<std::uint32_t>(groups[0])) {
    return;
  }
  const std::int32_t* group = groups + 1 + blockIdx.y * kGroupInts;
  const std::uint32_t expert = static_cast<std::uint32_t>(group[0]);
  // The table's own bound: a longer list would run into the next group's ints.
  const int count = min(group[1], static_cast<int>(kMaxGroupSlots));

  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int slice = wave % kSplit;
  const int row_group = wave / kSplit;
  const int t = lane % T::kTasks;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock +
                            row_group * kRowsPerWave + lane / T::kTasks;
  const std::uint32_t units = k / 256;
  const std::uint8_t* base =
      w + (static_cast<std::size_t>(expert) * m + row) * RowBytes<F>(k);

  for (int s0 = 0; s0 < count; s0 += kSlots) {
    float acc[kSlots];
    const float* xs[kSlots];
    int slot[kSlots];
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      acc[j] = 0.0F;
      slot[j] = s0 + j < count ? group[2 + s0 + j] : -1;
      const std::uint32_t r =
          slot[j] < 0 ? 0U
                      : static_cast<std::uint32_t>(slot[j]) / slots_per_row;
      xs[j] = x + static_cast<std::size_t>(r) * k;
    }
    if (row < m) {
      for (std::uint32_t unit = slice; unit < units; unit += kSplit) {
        float wv[T::kValues];
        std::uint32_t offset = 0;
        Decode<F>(base, unit, t, wv, &offset);
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            acc[j] = Contract<F>(acc[j], wv, xs[j] + offset);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      acc[j] = LaneSum<T::kTasks>(acc[j]);
    }
    if constexpr (kSplit == 1) {
      if (t == 0 && row < m) {
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            y[static_cast<std::size_t>(slot[j]) * m + row] = acc[j];
          }
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < kSlots; ++j) {
        partial[j][slice][row_group][lane] = acc[j];
      }
      __syncthreads();
      if (slice == 0 && t == 0 && row < m) {
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            float sum = partial[j][0][row_group][lane];
#pragma unroll
            for (int s = 1; s < kSplit; ++s) {
              sum += partial[j][s][row_group][lane];
            }
            y[static_cast<std::size_t>(slot[j]) * m + row] = sum;
          }
        }
      }
      __syncthreads();
    }
  }
}

template <WeightFormat F>
void LaunchGrouped(const GroupedGemvArgs& args, hipStream_t stream) {
  constexpr int kRowsPerWave = kWave / Traits<F>::kTasks;
  const auto* weights = static_cast<const std::uint8_t*>(args.w);
  const auto with_split = [&]<int kSplit>() {
    constexpr int kRows = kRowsPerWave * (kWaves / kSplit);
    const dim3 grid((args.m + kRows - 1) / kRows, args.max_groups);
    // A group lists one slot per row of the batch, so the launch's rows bound
    // the pass width. Wider passes only re-read an expert fewer times: a slot's
    // chain starts and ends in the same place at every width.
    const auto with_slots = [&]<int kSlots>() {
      GroupedGemvKernel<F, kSplit, kSlots>
          <<<grid, kThreads, 0, stream>>>(weights, args.groups, args.x, args.y,
                                          args.slots_per_row, args.m, args.k);
    };
    if (args.rows == 1) {
      with_slots.template operator()<1>();
    } else if (args.rows <= 4) {
      with_slots.template operator()<4>();
    } else {
      with_slots.template operator()<8>();
    }
  };
  // The split divides the row's 256-value super-blocks, so it is chosen from
  // their count: below four there is nothing to hand to a second wave, and the
  // widths this model uses (512, 2560, 6144) fall on either side of that.
  if (args.k / 256 < kSplitWaves) {
    with_split.template operator()<1>();
  } else {
    with_split.template operator()<kSplitWaves>();
  }
}

// ---------------------------------------------------------------------------
// Embedding gather
// ---------------------------------------------------------------------------

/// One row of the table per blockIdx.y, cut into the same lanes the projection
/// uses: a thread decodes 64 values and stores them where they belong, so the
/// gather needs no decoder of its own.
template <WeightFormat F>
__global__ void __launch_bounds__(kThreads) EmbedKernel(
    const std::uint8_t* __restrict__ table,
    const std::int32_t* __restrict__ tokens, float* __restrict__ y,
    std::uint32_t n, std::uint32_t lanes) {
  const std::uint32_t lane = blockIdx.x * kThreads + threadIdx.x;
  if (lane >= lanes) {
    return;
  }
  const std::uint32_t row = blockIdx.y;
  const std::uint32_t unit = lane / Traits<F>::kTasks;
  const int t = static_cast<int>(lane % Traits<F>::kTasks);
  const std::uint32_t token = static_cast<std::uint32_t>(tokens[row]);
  const std::uint8_t* base =
      table + static_cast<std::size_t>(token) * RowBytes<F>(n);
  float w[Traits<F>::kValues];
  std::uint32_t offset = 0;
  Decode<F>(base, unit, t, w, &offset);
  Scatter<F>(y + static_cast<std::size_t>(row) * n, w, offset);
}

/// Whether a format's rows stay word aligned at width k, which is what the
/// block loads round down to. Q4_K's row size is a multiple of 16 for any k the
/// blocks divide; Q6_K's is 210 bytes per block and needs an even block count.
template <WeightFormat F>
bool RowAligned(std::uint32_t k) noexcept {
  return F == WeightFormat::kQ4_K || (k / 256) % 2 == 0;
}


}  // namespace

bool GroupedGemv(const GroupedGemvArgs& args, hipStream_t stream) {
  if (args.w == nullptr || args.groups == nullptr || args.x == nullptr ||
      args.y == nullptr) {
    return false;
  }
  if (args.max_groups == 0 || args.m == 0 || args.slots_per_row == 0 ||
      args.rows == 0 || args.rows > kMaxGroupSlots) {
    return false;
  }
  if (args.k < 256 || args.k % 256 != 0) {
    return false;
  }
  // The decoders load sixteen bytes at a time out of a row, and the Q6_K ones
  // round an unaligned window down to a word: both need the tensor itself on a
  // multiple of 16.
  const auto bits = reinterpret_cast<std::uintptr_t>(args.w) |
                    reinterpret_cast<std::uintptr_t>(args.x);
  if ((bits & 15U) != 0) {
    return false;
  }
  switch (args.format) {
    case WeightFormat::kQ4_K:
      LaunchGrouped<WeightFormat::kQ4_K>(args, stream);
      return true;
    case WeightFormat::kQ6_K:
      if (!RowAligned<WeightFormat::kQ6_K>(args.k)) {
        return false;
      }
      LaunchGrouped<WeightFormat::kQ6_K>(args, stream);
      return true;
  }
  return false;
}

void EmbedRows(const EmbedArgs& args, hipStream_t stream) {
  if (args.table == nullptr || args.tokens == nullptr || args.y == nullptr) {
    throw std::invalid_argument(
        "kolibri1 embedding needs a table, its tokens and an output");
  }
  if (args.n < 256 || args.n % 256 != 0) {
    throw std::invalid_argument(
        "kolibri1 embedding needs a width the quantized blocks divide");
  }
  if ((reinterpret_cast<std::uintptr_t>(args.table) & 15U) != 0) {
    throw std::invalid_argument(
        "kolibri1 embedding needs a table on a 16-byte boundary");
  }
  if (args.rows == 0) {
    return;
  }
  const auto launch = [&]<WeightFormat F>() {
    const std::uint32_t lanes = (args.n / 256) * Traits<F>::kTasks;
    const dim3 grid((lanes + kThreads - 1) / kThreads, args.rows);
    EmbedKernel<F><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(args.table), args.tokens, args.y,
        args.n, lanes);
  };
  switch (args.format) {
    case WeightFormat::kQ4_K:
      launch.template operator()<WeightFormat::kQ4_K>();
      return;
    case WeightFormat::kQ6_K:
      if (!RowAligned<WeightFormat::kQ6_K>(args.n)) {
        throw std::invalid_argument(
            "kolibri1 embedding needs a Q6_K width of whole super-block pairs");
      }
      launch.template operator()<WeightFormat::kQ6_K>();
      return;
  }
  throw std::invalid_argument("kolibri1 embedding: unsupported weight format");
}

}  // namespace gufo::models::kolibri1::rocm
