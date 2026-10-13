// What a decode step's routed projections cost, and what limits them.
//
// The stage profile says the routed experts read 660 MB per decode token in
// 7.55 ms - 87 GB/s, where the same engine's output head reads 225 GB/s and its
// dense projections 170 GB/s. That gap is the decode lever, and it is a
// question about launch geometry rather than about the artifact: the routed
// projections of a layer are a 512x2560 gate, a 512x2560 up and a 2560x512 down
// over six experts with one slot each, and a step is fifty layers of that.
// Random weights move the same bytes through the same addresses, so the answer
// fits beside a running server in under a gigabyte.
//
// The kernel is Kolibri 1's grouped GEMV with its geometry turned into template
// arguments: the waves a row's super-blocks are shared between (kSplit), the
// block size, whether the unit loop keeps the next unit's bytes in flight while
// the current one is dequantized, and whether gate and up run as one launch over
// two tensors instead of two. The decoders are the engine's, copied - what is
// measured is a byte stream and it has to be the same byte stream.
//
// The first row of each shape is the launch the engine's rules pick, and it
// becomes the reference: a variant on the same split adds in the same order and
// must match it bit for bit - a variant that read the wrong bytes would move
// them just as fast, so that check is what keeps the sweep honest. A variant
// with a different split re-associates the sum, so it only has to stay within a
// rounding of it.
//
// Build on the box with the matrix cores:
//   hipcc -std=c++20 -O3 --offload-arch=gfx1151 \
//       tools/kolibri/decode_gemv_probe.cpp -o artifacts/kolibri/decode_gemv_probe
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK_HIP(call)                                                   \
  do {                                                                    \
    const hipError_t err__ = (call);                                      \
    if (err__ != hipSuccess) {                                            \
      std::fprintf(stderr, "hip: %s (%s:%d)\n", hipGetErrorString(err__), \
                   __FILE__, __LINE__);                                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace {

// ---------------------------------------------------------------------------
// The engine's decoders, verbatim (src/models/kolibri1/kernels/rocm/gemv.hip)
// ---------------------------------------------------------------------------

constexpr int kWave = 32;

__host__ __device__ __forceinline__ float Half(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__host__ __device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  const auto h = static_cast<unsigned short>(bits & 0xFFFFU);
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

__host__ __device__ __forceinline__ void ScaleMin(std::uint32_t w0,
                                                  std::uint32_t w1,
                                                  std::uint32_t w2, int j,
                                                  float* sc, float* m) {
  const std::uint32_t shift = 8U * static_cast<std::uint32_t>(j & 3);
  const std::uint32_t a = (w0 >> shift) & 0xFFU;
  const std::uint32_t b = (w1 >> shift) & 0xFFU;
  const std::uint32_t c = (w2 >> shift) & 0xFFU;
  const bool low = j < 4;
  *sc = static_cast<float>(low ? (a & 63U) : ((c & 0xFU) | ((a >> 6) << 4)));
  *m = static_cast<float>(low ? (b & 63U) : ((c >> 4) | ((b >> 6) << 4)));
}

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

__device__ __forceinline__ void Dequant16(const std::uint8_t* q, int shift,
                                          float scale, float offset,
                                          float* out) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t v = (q[i] >> shift) & 0xFU;
    out[i] = __builtin_fmaf(scale, static_cast<float>(v), offset);
  }
}

__device__ __forceinline__ void Dequant16Six(const std::uint8_t* q, int shift,
                                             const std::uint8_t* high,
                                             int hshift, float scale,
                                             float offset, float* out) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t low = (q[i] >> shift) & 0xFU;
    const std::uint32_t top = (high[i] >> hshift) & 0x30U;
    out[i] = __builtin_fmaf(scale, static_cast<float>(low | top), offset);
  }
}

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

enum Format { kQ4K, kQ6K };

template <Format F>
struct Traits;
template <>
struct Traits<kQ4K> {
  static constexpr int kBytes = 144;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
};
template <>
struct Traits<kQ6K> {
  static constexpr int kBytes = 210;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
};

template <Format F>
__host__ __device__ constexpr std::size_t RowBytes(std::uint32_t k) {
  return static_cast<std::size_t>(k) / 256 * Traits<F>::kBytes;
}

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
  Dequant16Six(ql_a, 0, qh, 0, s0, -32.0F * s0, w);
  Dequant16Six(ql_b, 0, qh, 2, s1, -32.0F * s1, w + 16);
  Dequant16Six(ql_a, 4, qh, 4, s2, -32.0F * s2, w + 32);
  Dequant16Six(ql_b, 4, qh, 6, s3, -32.0F * s3, w + 48);
}

template <Format F>
__device__ __forceinline__ void Decode(const std::uint8_t* row,
                                       std::uint32_t unit, int t,
                                       float (&w)[Traits<F>::kValues],
                                       std::uint32_t* offset) {
  const std::uint8_t* block =
      row + static_cast<std::size_t>(unit) * Traits<F>::kBytes;
  if constexpr (F == kQ6K) {
    DecodeQ6K(block, t, w);
    *offset = unit * 256 + 128 * static_cast<std::uint32_t>(t >> 1) +
              16 * static_cast<std::uint32_t>(t & 1);
  } else {
    DecodeQ4K(block, t, w);
    *offset = unit * 256 + 64 * static_cast<std::uint32_t>(t);
  }
}

template <Format F>
__device__ __forceinline__ float Contract(float acc,
                                          const float (&w)[Traits<F>::kValues],
                                          const float* x) {
  constexpr int kRuns = Traits<F>::kValues / 16;
  constexpr int kStride = F == kQ6K ? 32 : 16;
#pragma unroll
  for (int r = 0; r < kRuns; ++r) {
    acc = Accumulate<16>(acc, w + 16 * r, x + kStride * r);
  }
  return acc;
}

__device__ __forceinline__ float LaneSum(float acc) {
#pragma unroll
  for (int offset = 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
  }
  return acc;
}

/// A Q4_K unit out of registers instead of memory: the same three vectors the
/// decoder above loads, in the same order into the same slots of w.
__device__ __forceinline__ void ExpandQ4K(const uint4& header, const uint4& qa,
                                          const uint4& qb, int t,
                                          float (&w)[64]) {
  const float d = HalfBits(header.x);
  const float dmin = HalfBits(header.x >> 16);
  float sc0, m0, sc1, m1;
  ScaleMin(header.y, header.z, header.w, 2 * t, &sc0, &m0);
  ScaleMin(header.y, header.z, header.w, 2 * t + 1, &sc1, &m1);
  const auto* q0 = reinterpret_cast<const std::uint8_t*>(&qa);
  const auto* q1 = reinterpret_cast<const std::uint8_t*>(&qb);
  Dequant16(q0, 0, d * sc0, -(dmin * m0), w);
  Dequant16(q1, 0, d * sc0, -(dmin * m0), w + 16);
  Dequant16(q0, 4, d * sc1, -(dmin * m1), w + 32);
  Dequant16(q1, 4, d * sc1, -(dmin * m1), w + 48);
}

// ---------------------------------------------------------------------------
// The decode shape with its geometry as arguments
// ---------------------------------------------------------------------------

/// One slot per group - what a single token's six experts are. The chain is the
/// engine's: `unit = slice; unit += kSplit`, the butterfly inside the quad, the
/// split slices added in increasing order.
///
/// The prefetch is a Q4_K path: that format's unit is one aligned header and
/// two aligned payload vectors, so the next unit's three loads can go out
/// before this unit's arithmetic starts, which is the whole difference between
/// a dependent load chain and a streamed one. Q6_K reads three unaligned
/// windows per unit through the byte-alignment instruction; holding those over
/// an iteration costs registers and buys the same thing on a third of the
/// bytes, so the flag falls back to the plain loop there.
template <Format F, int kSplit, int kThreads, bool kPrefetch, bool kPair>
__global__ void __launch_bounds__(kThreads) DecodeGemvProbe(
    const std::uint8_t* __restrict__ w, const std::uint8_t* __restrict__ w2,
    const std::int32_t* __restrict__ groups, const float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ y2, std::uint32_t m,
    std::uint32_t k) {
  using T = Traits<F>;
  constexpr int kWaves = kThreads / kWave;
  constexpr int kRowGroups = kWaves / kSplit;
  constexpr int kRowsPerBlock = (kWave / T::kTasks) * kRowGroups;
  __shared__ float partial[kSplit][kRowGroups][kWave];

  if (blockIdx.y >= static_cast<std::uint32_t>(groups[0])) {
    return;
  }
  const std::int32_t* group = groups + 1 + blockIdx.y * 3;
  const std::uint32_t expert = static_cast<std::uint32_t>(group[0]);

  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int slice = wave % kSplit;
  const int row_group = wave / kSplit;
  const int t = lane % T::kTasks;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock +
                            row_group * (kWave / T::kTasks) + lane / T::kTasks;
  const std::uint32_t units = k / 256;
  const bool second = kPair && blockIdx.z == 1U;
  const std::uint8_t* base =
      (second ? w2 : w) +
      (static_cast<std::size_t>(expert) * m + row) * RowBytes<F>(k);
  float* out = second ? y2 : y;

  float acc = 0.0F;
  if (row < m) {
    if constexpr (kPrefetch && F == kQ4K) {
      const auto grab = [&](std::uint32_t unit, uint4& header, uint4& qa,
                            uint4& qb) {
        const std::uint8_t* block =
            base + static_cast<std::size_t>(unit) * T::kBytes;
        header = *reinterpret_cast<const uint4*>(block);
        qa = *reinterpret_cast<const uint4*>(block + 16 + 32 * t);
        qb = *reinterpret_cast<const uint4*>(block + 16 + 32 * t + 16);
      };
      uint4 header{}, qa{}, qb{};
      if (slice < units) {
        grab(slice, header, qa, qb);
      }
      for (std::uint32_t unit = slice; unit < units; unit += kSplit) {
        uint4 nh{}, nqa{}, nqb{};
        const std::uint32_t next = unit + kSplit;
        if (next < units) {
          grab(next, nh, nqa, nqb);  // issued before the chain below
        }
        float wv[64];
        ExpandQ4K(header, qa, qb, t, wv);
        acc = Contract<F>(acc, wv,
                          x + unit * 256 + 64 * static_cast<std::uint32_t>(t));
        header = nh;
        qa = nqa;
        qb = nqb;
      }
    } else {
      for (std::uint32_t unit = slice; unit < units; unit += kSplit) {
        float wv[T::kValues];
        std::uint32_t offset = 0;
        Decode<F>(base, unit, t, wv, &offset);
        acc = Contract<F>(acc, wv, x + offset);
      }
    }
  }
  acc = LaneSum(acc);
  if constexpr (kSplit == 1) {
    if (t == 0 && row < m) {
      out[static_cast<std::size_t>(blockIdx.y) * m + row] = acc;
    }
  } else {
    partial[slice][row_group][lane] = acc;
    __syncthreads();
    if (slice == 0 && t == 0 && row < m) {
      float sum = partial[0][row_group][lane];
#pragma unroll
      for (int s = 1; s < kSplit; ++s) {
        sum += partial[s][row_group][lane];
      }
      out[static_cast<std::size_t>(blockIdx.y) * m + row] = sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Shape {
  const char* name;
  Format format;
  std::uint32_t m;
  std::uint32_t k;
  std::uint32_t experts;  ///< matrices in the blob
  std::uint32_t touched;  ///< experts one step reads
};

struct Variant {
  const char* label;
  int split;
  int threads;
  bool prefetch;
  bool pair;
};

/// Launches per sweep, each on its own experts: the working set has to be
/// bigger than what the GPU can hold between two layers of a real step.
constexpr int kRotations = 32;

/// Rounds per variant, after one warm-up round; the minimum over rounds is the
/// number, because a sweep on a box that is also serving a model has no
/// reason to give the same launch the same memory controller twice.
constexpr int kRounds = 4;

struct Measured {
  double us = 0.0;       ///< 0 when the geometry does not fit the block
  std::size_t traffic = 0;
  double rel = 0.0;
  std::size_t diffs = 0;
};

/// Codes are random; the scale fields are float bits, so a block decodes to
/// plausible magnitudes instead of denormals.
template <Format F>
std::vector<std::uint8_t> MakeWeights(std::mt19937& rng, std::size_t rows,
                                      std::uint32_t k) {
  std::vector<std::uint8_t> blob(rows * RowBytes<F>(k));
  std::uniform_real_distribution<float> scale(0.001F, 0.02F);
  const std::size_t units = k / 256;
  // Word-wise, because a sweep allocates a few hundred of these megabytes and
  // a per-byte distribution would take longer than the measurement.
  std::uint32_t state = 0x9E3779B9U ^ static_cast<std::uint32_t>(rows * k);
  const auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t u = 0; u < units; ++u) {
      std::uint8_t* block =
          blob.data() + row * RowBytes<F>(k) + u * Traits<F>::kBytes;
      std::size_t i = 0;
      for (; i + 4 <= Traits<F>::kBytes; i += 4) {
        const std::uint32_t word = next();
        std::memcpy(block + i, &word, 4);
      }
      for (; i < Traits<F>::kBytes; ++i) {
        block[i] = static_cast<std::uint8_t>(next() & 0xFFU);
      }
      if constexpr (F == kQ4K) {
        const float d = scale(rng);
        std::uint16_t bits;
        std::memcpy(&bits, &d, 2);
        block[0] = static_cast<std::uint8_t>(bits & 0xFFU);
        block[1] = static_cast<std::uint8_t>(bits >> 8);
        block[2] = 0U;
        block[3] = 0U;
      } else {
        const float d = scale(rng);
        std::memcpy(block + 208, &d, 2);
      }
    }
  }
  return blob;
}
template <Format F, int kSplit, int kThreads, bool kPrefetch, bool kPair>
Measured Run(const Shape& s, const std::int32_t* groups_d, int table_ints,
             const std::uint8_t* w_d, const std::uint8_t* w2_d,
             const float* x_d, float* y_d, float* y2_d,
             const std::vector<float>& ref, std::size_t y_bytes) {
  constexpr int kWaves = kThreads / kWave;
  if (kSplit > kWaves) {  // a block cannot hold more slices than waves
    return Measured{};
  }
  constexpr int kRowGroups = kWaves / kSplit;
  constexpr int kRows = (kWave / Traits<F>::kTasks) * kRowGroups;
  const dim3 grid((s.m + kRows - 1) / kRows, s.touched, kPair ? 2 : 1);
  const std::size_t traffic =
      RowBytes<F>(s.k) * s.m * s.touched * (kPair ? 2 : 1);

  // A step touches six experts and will not see them again for fifty layers;
  // thirty launches over the same 4 MB would measure the L2, not the memory
  // the stage profile is talking about. Each rep runs on its own table, and
  // the tables cover disjoint experts.
  const auto launch = [&](int rep) {
    const std::int32_t* table = groups_d +
                                static_cast<std::size_t>(rep % kRotations) *
                                    static_cast<std::size_t>(table_ints);
    DecodeGemvProbe<F, kSplit, kThreads, kPrefetch, kPair><<<grid, kThreads>>>(
        w_d, w2_d, table, x_d, y_d, y2_d, s.m, s.k);
  };
  launch(0);
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipDeviceSynchronize());

  hipEvent_t first, last;
  CHECK_HIP(hipEventCreate(&first));
  CHECK_HIP(hipEventCreate(&last));
  constexpr int kReps = kRotations * 2;
  CHECK_HIP(hipEventRecord(first));
  for (int i = 0; i < kReps; ++i) {
    launch(i);
  }
  CHECK_HIP(hipEventRecord(last));
  CHECK_HIP(hipEventSynchronize(last));
  float ms = 0.0F;
  CHECK_HIP(hipEventElapsedTime(&ms, first, last));
  CHECK_HIP(hipEventDestroy(first));
  CHECK_HIP(hipEventDestroy(last));

  Measured out;
  out.us = ms * 1000.0 / kReps;
  out.traffic = traffic;

  // The rotations share one output, so put rotation 0's answer back before
  // reading: that is the table the reference was captured from.
  launch(0);
  CHECK_HIP(hipDeviceSynchronize());
  const std::size_t count = y_bytes / sizeof(float);
  std::vector<float> got(count);
  CHECK_HIP(hipMemcpy(got.data(), y_d, y_bytes, hipMemcpyDeviceToHost));
  if (!ref.empty()) {
    for (std::size_t i = 0; i < count; ++i) {
      const double want = static_cast<double>(ref[i]);
      const double d = std::fabs(static_cast<double>(got[i]) - want);
      if (d > 0.0) {
        ++out.diffs;
      }
      out.rel = std::max(out.rel, d / std::max(1.0, std::fabs(want)));
    }
  }
  return out;
}

/// One variant, dispatching the template arguments the table asked for.
Measured Dispatch(const Shape& s, const Variant& v,
                  const std::int32_t* groups_d, int table_ints,
                  const std::uint8_t* w_d, const std::uint8_t* w2_d,
                  const float* x_d, float* y_d, float* y2_d,
                  const std::vector<float>& ref, std::size_t y_bytes) {
  const auto pick = [&]<Format F, int kSplit, int kThreads, bool kPre,
                        bool kPr>() -> Measured {
    return Run<F, kSplit, kThreads, kPre, kPr>(s, groups_d, table_ints, w_d,
                                               w2_d, x_d, y_d, y2_d, ref,
                                               y_bytes);
  };
  const auto by_split = [&]<int kSplit>() -> Measured {
    const auto by_threads = [&]<int kThreads>() -> Measured {
      if (v.prefetch && v.pair) {
        return s.format == kQ4K
                   ? pick.template operator()<kQ4K, kSplit, kThreads, true,
                                             true>()
                   : pick.template operator()<kQ6K, kSplit, kThreads, true,
                                             true>();
      }
      if (v.prefetch) {
        return s.format == kQ4K
                   ? pick.template operator()<kQ4K, kSplit, kThreads, true,
                                             false>()
                   : pick.template operator()<kQ6K, kSplit, kThreads, true,
                                             false>();
      }
      if (v.pair) {
        return s.format == kQ4K
                   ? pick.template operator()<kQ4K, kSplit, kThreads, false,
                                             true>()
                   : pick.template operator()<kQ6K, kSplit, kThreads, false,
                                             true>();
      }
      return s.format == kQ4K
                 ? pick.template operator()<kQ4K, kSplit, kThreads, false,
                                           false>()
                 : pick.template operator()<kQ6K, kSplit, kThreads, false,
                                           false>();
    };
    switch (v.threads) {
      case 128:
        return by_threads.template operator()<128>();
      case 512:
        return by_threads.template operator()<512>();
      default:
        return by_threads.template operator()<256>();
    }
  };
  switch (v.split) {
    case 1:
      return by_split.template operator()<1>();
    case 2:
      return by_split.template operator()<2>();
    case 8:
      return by_split.template operator()<8>();
    case 16:
      return by_split.template operator()<16>();
    default:
      return by_split.template operator()<4>();
  }
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::fprintf(stderr, "this probe measures the Strix Halo memory path\n");
    return 1;
  }
  std::printf("device %s, %d CUs\n", prop.name, prop.multiProcessorCount);
  std::printf("Every variant runs %d rounds on the rotating tables and keeps its "
              "best launch;\nthe first round is warm-up. `rel` is the worst "
              "|variant - reference|\n/ max(1, |reference|) at the end; a "
              "variant on the reference's split must\nread 0.00e+00 in 0 rows.\n\n",
              std::to_string(kRounds + 1).c_str());

  std::mt19937 rng(20261013);
  std::uniform_real_distribution<float> unit(-1.0F, 1.0F);

  const Shape shapes[] = {
      {"dense qkv       m=6144 k=2560  Q4_K", kQ4K, 6144, 2560, kRotations, 1},
      {"routed gate/up  m=512  k=2560  Q4_K", kQ4K, 512, 2560, kRotations * 6,
       6},
      {"routed down     m=2560 k=512   Q4_K", kQ4K, 2560, 512, kRotations * 6,
       6},
      {"routed down     m=2560 k=512   Q6_K", kQ6K, 2560, 512, kRotations * 6,
       6},
  };

  for (const Shape& s : shapes) {
    // One table per rotation: the experts a rotation reads are its own, so the
    // sweep reaches over kRotations launches of memory instead of one.
    const int table_ints = 1 + static_cast<int>(s.touched) * 3;
    std::vector<std::int32_t> h_groups(
        static_cast<std::size_t>(kRotations) * table_ints, 0);
    for (int r = 0; r < kRotations; ++r) {
      std::int32_t* table = h_groups.data() + r * table_ints;
      table[0] = static_cast<std::int32_t>(s.touched);
      for (std::uint32_t g = 0; g < s.touched; ++g) {
        table[1 + g * 3 + 0] = static_cast<std::int32_t>(r * s.touched + g);
        table[1 + g * 3 + 1] = 1;
        table[1 + g * 3 + 2] = static_cast<std::int32_t>(g);
      }
    }
    std::int32_t* groups_d = nullptr;
    CHECK_HIP(hipMalloc(&groups_d, h_groups.size() * sizeof(std::int32_t)));
    CHECK_HIP(hipMemcpy(groups_d, h_groups.data(),
                        h_groups.size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));

    std::vector<float> x(static_cast<std::size_t>(s.k));
    for (float& v : x) {
      v = unit(rng);
    }
    float* x_d = nullptr;
    CHECK_HIP(hipMalloc(&x_d, x.size() * sizeof(float)));
    CHECK_HIP(hipMemcpy(x_d, x.data(), x.size() * sizeof(float),
                        hipMemcpyHostToDevice));

    const std::size_t rows = static_cast<std::size_t>(s.experts) * s.m;
    std::vector<std::uint8_t> blob_a, blob_b;
    if (s.format == kQ4K) {
      blob_a = MakeWeights<kQ4K>(rng, rows, s.k);
      blob_b = MakeWeights<kQ4K>(rng, rows, s.k);
    } else {
      blob_a = MakeWeights<kQ6K>(rng, rows, s.k);
      blob_b = MakeWeights<kQ6K>(rng, rows, s.k);
    }
    std::uint8_t *w_d = nullptr, *w2_d = nullptr;
    CHECK_HIP(hipMalloc(&w_d, blob_a.size()));
    CHECK_HIP(hipMalloc(&w2_d, blob_b.size()));
    CHECK_HIP(hipMemcpy(w_d, blob_a.data(), blob_a.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w2_d, blob_b.data(), blob_b.size(), hipMemcpyHostToDevice));

    const std::size_t y_bytes =
        static_cast<std::size_t>(s.touched) * s.m * sizeof(float);
    float* y_d = nullptr;
    float* y2_d = nullptr;
    CHECK_HIP(hipMalloc(&y_d, y_bytes));
    CHECK_HIP(hipMalloc(&y2_d, y_bytes));

    const int chosen = (s.k / 256 >= 4) ? 4 : 1;
    const Variant variants[] = {
        {"launcher's choice", chosen, 256, false, false},
        {"split 1, 256", 1, 256, false, false},
        {"split 2, 256", 2, 256, false, false},
        {"split 4, 256", 4, 256, false, false},
        {"split 8, 256", 8, 256, false, false},
        {"split 2, 128", 2, 128, false, false},
        {"split 4, 128", 4, 128, false, false},
        {"split 4, 512", 4, 512, false, false},
        {"split 8, 512", 8, 512, false, false},
        {"split 16, 512", 16, 512, false, false},
        {"reference + prefetch", chosen, 256, true, false},
        {"split 2, 256, prefetch", 2, 256, true, false},
        {"split 2, 256, gate+up", 2, 256, false, true},
        {"split 4, 256, gate+up", 4, 256, false, true},
        {"split 4, 512, gate+up", 4, 512, false, true},
        {"split 4, 256, prefetch, gate+up", 4, 256, true, true},
    };
    const std::size_t n = sizeof(variants) / sizeof(variants[0]);

    const double per_launch =
        static_cast<double>(s.format == kQ4K ? 144 : 210) *
        static_cast<double>(s.k) / 256.0 * static_cast<double>(s.m) *
        s.touched / (1024.0 * 1024.0);
    std::printf("%s  -  %.2f MB per launch, %d rotations = %.0f MB per round\n",
                s.name, per_launch, kRotations, per_launch * kRotations);

    std::vector<float> ref;
    std::vector<double> best(n, 0.0);
    std::vector<Measured> last(n);
    for (int round = 0; round <= kRounds; ++round) {
      for (std::size_t i = 0; i < n; ++i) {
        Measured m = Dispatch(s, variants[i], groups_d, table_ints, w_d, w2_d,
                              x_d, y_d, y2_d, ref, y_bytes);
        if (round == 0 && i == 0) {
          ref.resize(y_bytes / sizeof(float));
          CHECK_HIP(
              hipMemcpy(ref.data(), y_d, y_bytes, hipMemcpyDeviceToHost));
        }
        if (round == 0 || m.us <= 0.0) {
          last[i] = m;
          continue;
        }
        if (best[i] <= 0.0 || m.us < best[i]) {
          best[i] = m.us;
        }
        last[i] = m;
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      if (best[i] <= 0.0) {
        std::printf("  %-36s skipped (%d slices in %d threads)\n",
                    variants[i].label, variants[i].split, variants[i].threads);
        continue;
      }
      const double gb_s = static_cast<double>(last[i].traffic) / (best[i] * 1e3);
      if (i == 0) {
        std::printf("  %-36s %8.2f us %8.1f GB/s   (reference)\n",
                    variants[i].label, best[i], gb_s);
      } else {
        std::printf(
            "  %-36s %8.2f us %8.1f GB/s  rel %.2e in %zu of %zu rows\n",
            variants[i].label, best[i], gb_s, last[i].rel, last[i].diffs,
            y_bytes / sizeof(float));
      }
    }
    std::printf("\n");

    CHECK_HIP(hipFree(y2_d));
    CHECK_HIP(hipFree(y_d));
    CHECK_HIP(hipFree(w2_d));
    CHECK_HIP(hipFree(w_d));
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(groups_d));
  }
  return 0;
}
