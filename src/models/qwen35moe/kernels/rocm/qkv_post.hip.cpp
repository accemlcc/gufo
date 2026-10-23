// Qwen3.5-MoE's full-attention post projection: per-head Q/K RMSNorm, the
// rotary prefix turn, the packed gate halves laid out beside the queries, and
// the K/V cache write, in one launch so a row crosses memory once.
//
// The packed projection interleaves query and gate per head: head h owns the
// two `dim` slices at [2 h dim, (2 h + 2) dim), query first. A flat
// halves split would mix queries of one half of the heads with gates of the
// other; the lanes below never form one.
//
// One wave per (row, slot): slots [0, heads) are queries (which also publish
// their gate half), [heads, heads + kv_heads) keys, the rest values. A block
// that runs past the last slot keeps its threads to no barrier and returns,
// like the kolibri kernel this follows. Lanes hold their eight values
// strided, so a row's norm is one wave sum and a rotary pair always sits in
// one lane.
#include "src/models/qwen35moe/kernels/rocm/qkv_post.hpp"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gufo::models::qwen35moe::rocm {
namespace {

constexpr std::uint32_t kWave = 32;
constexpr std::uint32_t kWaves = 8;
constexpr std::uint32_t kThreads = kWave * kWaves;
/// Values per lane for a 256-wide head at a stride of one wave.
constexpr std::uint32_t kPerLane = 8;

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

__global__ void __launch_bounds__(kThreads)
    QkvPostKernel(QkvPostArgs a) {
  __shared__ float s_cos[32];
  __shared__ float s_sin[32];
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t wave = threadIdx.x / kWave;
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t slots = a.heads + 2 * a.kv_heads;
  const std::uint32_t slot = blockIdx.y * kWaves + wave;
  const bool active = row < a.rows && slot < slots;
  const std::uint32_t position = a.first_position + row;
  if (position >= a.max_context) {
    return;  // the caller sizes the cache; writing past it faults
  }

  // The turn is formed in double and only the result is shared: one entry
  // per rotary pair, thirty-two pairs.
  if (threadIdx.x < a.rotary_dim / 2) {
    const double ang = static_cast<double>(position) *
                       pow(a.theta_scale, static_cast<double>(threadIdx.x));
    double c = 0.0, s = 0.0;
    sincos(ang, &s, &c);
    s_cos[threadIdx.x] = static_cast<float>(c);
    s_sin[threadIdx.x] = static_cast<float>(s);
  }
  __syncthreads();

  if (!active) {
    return;  // an idle wave of the last block, sharing no barrier with rest
  }
  const bool is_q = slot < a.heads;
  const bool is_k = !is_q && slot < a.heads + a.kv_heads;
  const std::uint32_t h = is_q ? slot : 0;
  const std::uint32_t kh =
      is_k ? slot - a.heads : (!is_q ? slot - a.heads - a.kv_heads : 0);
  const std::uint32_t qd = a.heads * a.dim;

  float xh[kPerLane];
  if (is_q) {
    // Deinterleaved query half of head h, strided across the lanes.
    const float* src = a.qg + static_cast<std::size_t>(row) * 2 * qd +
                       static_cast<std::size_t>(h) * 2 * a.dim;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      xh[j] = src[lane + j * kWave];
    }
    float ss = 0.0F;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      ss += xh[j] * xh[j];
    }
    const float mean = WaveSum(ss) / static_cast<float>(a.dim);
    const float r = 1.0F / sqrtf(mean + a.eps);
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      xh[j] = xh[j] * r * a.q_norm[lane + j * kWave];
    }
    // NeoX halves over the rotary prefix: pair (i, i + 32) sits in lane i.
    if (lane < a.rotary_dim / 2) {
      const float c = s_cos[lane];
      const float s = s_sin[lane];
      const float p = xh[0];
      const float q = xh[1];
      xh[0] = p * c - q * s;
      xh[1] = p * s + q * c;
    }
    float* dst =
        a.q_out + (static_cast<std::size_t>(row) * a.heads + h) * a.dim;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      dst[lane + j * kWave] = xh[j];
    }
    // The gate half beside it, untouched.
    const float* gsrc = a.qg + static_cast<std::size_t>(row) * 2 * qd +
                        static_cast<std::size_t>(h) * 2 * a.dim + a.dim;
    float* gdst =
        a.gate_out + (static_cast<std::size_t>(row) * a.heads + h) * a.dim;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      gdst[lane + j * kWave] = gsrc[lane + j * kWave];
    }
    return;
  }
  if (is_k) {
    const float* src = a.k + static_cast<std::size_t>(row) * a.kv_heads *
                                  a.dim +
                       static_cast<std::size_t>(kh) * a.dim;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      xh[j] = src[lane + j * kWave];
    }
    float ss = 0.0F;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      ss += xh[j] * xh[j];
    }
    const float mean = WaveSum(ss) / static_cast<float>(a.dim);
    const float r = 1.0F / sqrtf(mean + a.eps);
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      xh[j] = xh[j] * r * a.k_norm[lane + j * kWave];
    }
    if (lane < a.rotary_dim / 2) {
      const float c = s_cos[lane];
      const float s = s_sin[lane];
      const float p = xh[0];
      const float q = xh[1];
      xh[0] = p * c - q * s;
      xh[1] = p * s + q * c;
    }
    float* dst = a.k_cache + (static_cast<std::size_t>(kh) * a.max_context +
                              position) *
                                 a.dim;
    for (std::uint32_t j = 0; j < kPerLane; ++j) {
      dst[lane + j * kWave] = xh[j];
    }
    return;
  }
  // Values: no norm, no turn, straight into the cache.
  const float* vsrc = a.v + static_cast<std::size_t>(row) * a.kv_heads * a.dim +
                      static_cast<std::size_t>(kh) * a.dim;
  float* vdst = a.v_cache + (static_cast<std::size_t>(kh) * a.max_context +
                             position) *
                                a.dim;
  for (std::uint32_t j = 0; j < kPerLane; ++j) {
    vdst[lane + j * kWave] = vsrc[lane + j * kWave];
  }
}

}  // namespace

void QkvPost(const QkvPostArgs& args, hipStream_t stream) {
  if (args.dim != 256) {
    throw std::invalid_argument("QkvPost supports a 256-wide head only");
  }
  if (args.rotary_dim != 64) {
    throw std::invalid_argument("QkvPost supports a rotary prefix of 64");
  }
  if (args.rows == 0) {
    return;
  }
  if (args.first_position + args.rows > args.max_context) {
    throw std::invalid_argument("QkvPost writes past the KV cache");
  }
  const std::uint32_t slots = args.heads + 2 * args.kv_heads;
  const dim3 grid(args.rows, (slots + kWaves - 1) / kWaves);
  QkvPostKernel<<<grid, kThreads, 0, stream>>>(args);
}

}  // namespace gufo::models::qwen35moe::rocm
