#include "src/models/kolibri1/kernels/rocm/qkv_post.hpp"

#include <hip/hip_fp16.h>

#include <stdexcept>

namespace gufo::models::kolibri1::rocm {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kWave = 32;
constexpr std::uint32_t kWaves = 8;
constexpr std::uint32_t kThreads = kWave * kWaves;
/// Four dimensions per lane. That puts the partner of dimension d, which is
/// d + 64, sixteen lanes away in the same slot of its float4, so one shuffle
/// reaches every pair.
constexpr std::uint32_t kPerLane = kDim / kWave;

/// v from lane (lane ^ M) through DPP or a cross-row permute: this is a
/// register-to-register move, where a shuffle is a round trip.
template<int M>
__device__ __forceinline__ float XorLane(float v) {
  static_assert(M > 0 && M < kWave && (M & (M - 1)) == 0);
  const int x = __float_as_int(v);
  if constexpr (M == 16) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        x, x, 0x76543210, 0xfedcba98, false, false));
  } else {
    // row_xmask:M
    return __int_as_float(
        __builtin_amdgcn_update_dpp(x, x, 0x160 | M, 0xf, 0xf, false));
  }
}

__device__ __forceinline__ float WaveSum(float v) {
  v += XorLane<16>(v);
  v += XorLane<8>(v);
  v += XorLane<4>(v);
  v += XorLane<2>(v);
  return v + XorLane<1>(v);
}

__device__ __forceinline__ float4 XorLane16(float4 v) {
  return {XorLane<16>(v.x), XorLane<16>(v.y), XorLane<16>(v.z),
          XorLane<16>(v.w)};
}

/// One wave per (row, head): normalize the head, turn it, store it.
///
/// A block that runs past the last head keeps its threads to nothing but the
/// barriers. Returning early instead would leave a barrier with fewer threads,
/// which is undefined rather than merely wasteful, and the last block of the
/// grid holds both key and value heads so it is rarely full.
__global__ void __launch_bounds__(kThreads) QkvPostKernel(QkvPostArgs a) {
  __shared__ float s_cos[kDim / 2];
  __shared__ float s_sin[kDim / 2];
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t wave = threadIdx.x / kWave;
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y * kWaves + wave;
  const std::uint32_t total = a.heads + 2 * a.kv_heads;
  const bool active = head < total && row < a.rows;
  const bool query = active && head < a.heads;
  const bool key = active && !query && head < a.heads + a.kv_heads;
  const std::uint32_t kv_head =
      active && !query ? head - a.heads - (key ? 0U : a.kv_heads) : 0U;
  const std::uint32_t position = a.first_position + row;

  // One thread per rotation pair. The turn is formed in double and only the
  // result is shared, so the cost is sixty-four double sines per block instead
  // of one per head and dimension.
  if (a.apply_rope && active && threadIdx.x < kDim / 2) {
    const double angle = static_cast<double>(position) *
                         pow(a.theta_scale, static_cast<double>(threadIdx.x));
    double c = 0.0;
    double s = 0.0;
    sincos(angle, &s, &c);
    s_cos[threadIdx.x] = static_cast<float>(c);
    s_sin[threadIdx.x] = static_cast<float>(s);
  }
  if (a.apply_rope) {
    __syncthreads();
  }

  const std::size_t head_row =
      query ? static_cast<std::size_t>(row) * a.heads + head
            : static_cast<std::size_t>(row) * a.kv_heads + kv_head;
  const std::size_t dim0 = static_cast<std::size_t>(lane) * kPerLane;
  const float* src = (query ? a.q : key ? a.k : a.v) + head_row * kDim + dim0;
  float4 x = {0.0F, 0.0F, 0.0F, 0.0F};
  if (active) {
    x = *reinterpret_cast<const float4*>(src);
  }
  if (!query && !key) {
    // Values are neither normed nor turned; they only change width and move.
    if (active) {
      const std::size_t slot = a.ring != 0 ? position % a.ring : position;
      __half* dst = reinterpret_cast<__half*>(
          a.v_cache + slot * static_cast<std::size_t>(a.kv_heads) * kDim +
          static_cast<std::size_t>(kv_head) * kDim + dim0);
      const float out[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (std::uint32_t j = 0; j < kPerLane; ++j) {
        dst[j] = __float2half_rn(out[j]);
      }
    }
    return;
  }

  const float r =
      rsqrtf(WaveSum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w) *
                 (1.0F / static_cast<float>(kDim)) +
             a.eps);
  const float* w = query ? a.q_norm : a.k_norm;
  x.x *= r * w[dim0 + 0];
  x.y *= r * w[dim0 + 1];
  x.z *= r * w[dim0 + 2];
  x.w *= r * w[dim0 + 3];

  if (a.apply_rope) {
    // Lanes below sixteen hold dimension d and read their partner d + 64 from
    // lane + 16; the lanes above hold the partner and need the same angle, so
    // the index wraps and only the sign of the cross term differs.
    const float4 p = XorLane16(x);
    const std::uint32_t i0 = (lane % (kWave / 2)) * kPerLane;
    const float turn = lane < kWave / 2 ? -1.0F : 1.0F;
    x.x = x.x * s_cos[i0 + 0] + turn * p.x * s_sin[i0 + 0];
    x.y = x.y * s_cos[i0 + 1] + turn * p.y * s_sin[i0 + 1];
    x.z = x.z * s_cos[i0 + 2] + turn * p.z * s_sin[i0 + 2];
    x.w = x.w * s_cos[i0 + 3] + turn * p.w * s_sin[i0 + 3];
  }

  if (!active) {
    return;
  }
  if (query) {
    *reinterpret_cast<float4*>(a.q + head_row * kDim + dim0) = x;
    return;
  }
  const std::size_t slot = a.ring != 0 ? position % a.ring : position;
  __half* dst = reinterpret_cast<__half*>(
      a.k_cache + slot * static_cast<std::size_t>(a.kv_heads) * kDim +
      static_cast<std::size_t>(kv_head) * kDim + dim0);
  const float out[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
  for (std::uint32_t j = 0; j < kPerLane; ++j) {
    dst[j] = __float2half_rn(out[j]);
  }
}

}  // namespace

void QkvPost(const QkvPostArgs& args, hipStream_t stream) {
  if (args.head_dim != kDim) {
    throw std::invalid_argument("QkvPost supports head_dim 128");
  }
  if (args.heads % args.kv_heads != 0) {
    throw std::invalid_argument(
        "QkvPost needs a whole number of query heads per key head");
  }
  if (args.rows == 0) {
    return;
  }
  const std::uint32_t total = args.heads + 2 * args.kv_heads;
  const dim3 grid(args.rows, (total + kWaves - 1) / kWaves);
  QkvPostKernel<<<grid, kThreads, 0, stream>>>(args);
}

}  // namespace gufo::models::kolibri1::rocm
