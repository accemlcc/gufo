// Qwen3.5-MoE's full-attention decode: online-softmax grouped-query attention
// over the F32 key/value cache. One wave per (row, query head) walks the
// cached keys of its group once for the online softmax statistics and a second
// time against the values, so a row crosses the keys twice and nothing else.
//
// This is the correct-but-plain shape of the path: no tiling, no split-K,
// F32 throughout. The production shape (tiles, matrix cores) keeps these
// numbers, and this file is what it is checked against.
#include "src/models/qwen35moe/kernels/rocm/attention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gufo::models::qwen35moe::rocm {
namespace {

constexpr std::uint32_t kWave = 32;
constexpr std::uint32_t kWaves = 8;
constexpr std::uint32_t kThreads = kWave * kWaves;

template<int M>
__device__ __forceinline__ float XorLane(float v) {
  static_assert(M > 0 && M < kWave && (M & (M - 1)) == 0);
  const int x = __float_as_int(v);
  return __int_as_float(__builtin_amdgcn_update_dpp(x, x, 0x160 | M, 0xf, 0xf,
                                                   false));
}

__device__ __forceinline__ float WaveSum(float v) {
  v += XorLane<16>(v);
  v += XorLane<8>(v);
  v += XorLane<4>(v);
  v += XorLane<2>(v);
  return v + XorLane<1>(v);
}

__device__ __forceinline__ float WaveMax(float v) {
  v = fmaxf(v, XorLane<16>(v));
  v = fmaxf(v, XorLane<8>(v));
  v = fmaxf(v, XorLane<4>(v));
  v = fmaxf(v, XorLane<2>(v));
  return fmaxf(v, XorLane<1>(v));
}

/// One wave per (row, query head). Lanes split the head dim; every lane holds
/// its strided share of q and of the accumulator across both key passes.
__global__ void __launch_bounds__(kThreads) AttentionKernel(AttentionArgs a) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t wave = threadIdx.x / kWave;
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t h = blockIdx.y * kWaves + wave;
  if (row >= a.rows || h >= a.heads) {
    return;  // an idle wave of the last block, sharing no barrier with rest
  }
  const std::uint32_t group = a.heads / a.kv_heads;
  const std::uint32_t kvh = h / group;
  const std::uint32_t hi = a.first_position + row + 1;
  const float* qh = a.q + (static_cast<std::size_t>(row) * a.heads + h) * a.dim;

  constexpr std::uint32_t kPerLane = 8;  // dim 256 over 32 lanes
  float qv[kPerLane];
  for (std::uint32_t j = 0; j < kPerLane; ++j) {
    qv[j] = qh[lane + j * kWave];
  }

  // First pass: online softmax statistics over the cached keys.
  float best = -INFINITY;
  float total = 0.0F;
  for (std::uint32_t p = 0; p < hi; ++p) {
    const float* krow = a.k_cache + (static_cast<std::size_t>(kvh) *
                                          a.max_context +
                                      p) *
                                         a.dim;
    float s = 0.0F;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      s += qv[j] * krow[lane + j * kWave];
    }
    // The butterfly leaves the full dot in every lane, so no broadcast
    // is needed before the online update below.
    s = WaveSum(s) * a.scale;
    const float nb = fmaxf(best, s);
    total = total * expf(best - nb) + expf(s - nb);
    best = nb;
  }

  // Second pass: accumulate exp(score - best) * value.
  float acc[kPerLane] = {};
  for (std::uint32_t p = 0; p < hi; ++p) {
    const float* krow = a.k_cache + (static_cast<std::size_t>(kvh) *
                                          a.max_context +
                                      p) *
                                         a.dim;
    float s = 0.0F;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      s += qv[j] * krow[lane + j * kWave];
    }
    s = WaveSum(s) * a.scale;
    const float w = expf(s - best) / total;
    const float* vrow = a.v_cache + (static_cast<std::size_t>(kvh) *
                                          a.max_context +
                                      p) *
                                         a.dim;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      acc[j] += w * vrow[lane + j * kWave];
    }
  }
  float* out = a.out + (static_cast<std::size_t>(row) * a.heads + h) * a.dim;
  for (std::uint32_t j = 0; j < kPerLane; ++j) {
    out[lane + j * kWave] = acc[j];
  }
}

}  // namespace

void Attention(const AttentionArgs& args, hipStream_t stream) {
  if (args.dim != 256 || args.heads % args.kv_heads != 0) {
    throw std::invalid_argument(
        "Attention needs a 256-wide head and a divisible GQA group");
  }
  if (args.rows == 0) {
    return;
  }
  if (args.first_position + args.rows > args.max_context) {
    throw std::invalid_argument("Attention reads past the KV cache");
  }
  const dim3 grid(args.rows, (args.heads + kWaves - 1) / kWaves);
  AttentionKernel<<<grid, kThreads, 0, stream>>>(args);
}

}  // namespace gufo::models::qwen35moe::rocm
