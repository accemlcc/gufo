#include <cstdint>
#include <stdexcept>

#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen35moe::rocm {
namespace {

constexpr int kWave = 32;
/// Threads per block. A narrow row does not feed all of them, so those shapes
/// run one wave per row and a block covers eight rows.
constexpr int kBlock = 256;
/// Floats per vector load. A row is a multiple of 32 floats, so a row boundary
/// is never inside a vector and every row starts on a vector boundary.
constexpr std::uint32_t kVec = 4;
/// float4 a lane holds for its row, per shape. These two numbers are the whole
/// supported width: a row a lane cannot hold in registers is refused by the
/// launcher rather than read from memory twice.
constexpr int kMaxWaveWindow = 4;
constexpr int kMaxBlockWindow = 8;

[[nodiscard]] constexpr std::uint32_t Groups(std::uint32_t n) noexcept {
  return n / kVec;
}

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

/// Sum over a row held by `kLanes` threads, in an order set by `kLanes` and by
/// nothing else. The waves add in ascending index, so a row keeps its value
/// when a batch boundary moves it into another block.
template<int kLanes>
__device__ __forceinline__ float RowSum(float v, float* scratch) {
  v = WaveSum(v);
  if constexpr (kLanes > kWave) {
    if (threadIdx.x % kWave == 0) {
      scratch[threadIdx.x / kWave] = v;
    }
    __syncthreads();
    float total = scratch[0];
#pragma unroll
    for (int w = 1; w < kLanes / kWave; ++w) {
      total += scratch[w];
    }
    v = total;
  }
  return v;
}

__device__ __forceinline__ float4 Zero4() {
  return {0.0F, 0.0F, 0.0F, 0.0F};
}

/// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, with w optional.
///
/// One row per wave while the row is narrow and one per block beyond, each lane
/// keeping its part of the row in registers so it crosses memory once. The
/// scale is the correctly rounded reciprocal square root rather than the fast
/// approximation: it costs one division per row, and it is what puts a
/// normalized row within an ulp of the reference instead of within a few.
template<int kLanes, int kWindow>
__global__ void __launch_bounds__(kBlock) RmsNormKernel(NormRowsArgs a) {
  constexpr int kRowsPerBlock = kBlock / kLanes;
  [[maybe_unused]] __shared__ float
      scratch[kLanes > kWave ? kLanes / kWave : 1];
  const std::uint32_t lane = threadIdx.x % kLanes;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * kRowsPerBlock +
                          threadIdx.x / kLanes;
  if (kLanes == kWave && row >= a.rows) {
    return;  // an idle wave of the last block, sharing no barrier with the rest
  }
  const std::uint32_t groups = Groups(a.n);
  const float4* xr = reinterpret_cast<const float4*>(a.x) +
                     static_cast<std::size_t>(row) * groups;
  float4 xv[kWindow];
  float ss = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kWindow; ++j) {
    const std::uint32_t g = lane + j * kLanes;
    xv[j] = g < groups ? xr[g] : Zero4();
    ss += xv[j].x * xv[j].x;
    ss += xv[j].y * xv[j].y;
    ss += xv[j].z * xv[j].z;
    ss += xv[j].w * xv[j].w;
  }
  const float mean = RowSum<kLanes>(ss, scratch) / static_cast<float>(a.n);
  const float r = 1.0F / sqrtf(mean + a.eps);
  float4* yr =
      reinterpret_cast<float4*>(a.y) + static_cast<std::size_t>(row) * groups;
  const float4* wr =
      a.w == nullptr ? nullptr : reinterpret_cast<const float4*>(a.w);
#pragma unroll
  for (std::uint32_t j = 0; j < kWindow; ++j) {
    const std::uint32_t g = lane + j * kLanes;
    if (g >= groups) {
      break;
    }
    float4 o = {xv[j].x * r, xv[j].y * r, xv[j].z * r, xv[j].w * r};
    if (wr != nullptr) {
      const float4 w = wr[g];
      o.x *= w.x;
      o.y *= w.y;
      o.z *= w.z;
      o.w *= w.w;
    }
    yr[g] = o;
  }
}

/// x[i] += add[i] over whole vectors. The elements past the last whole vector
/// of a count that is not a multiple of four are added scalar by the first
/// threads of the first block rather than by padding the buffer.
__global__ void __launch_bounds__(kBlock)
    AddKernel(float* x, const float* add, std::size_t groups,
              std::size_t count) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * kBlock + threadIdx.x;
  float4* xv = reinterpret_cast<float4*>(x);
  const float4* av = reinterpret_cast<const float4*>(add);
  if (first < groups) {
    const float4 a = xv[first];
    const float4 b = av[first];
    xv[first] = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  }
  const std::size_t tail = groups * kVec;
  if (first < count - tail) {
    x[tail + first] += add[tail + first];
  }
}

/// silu(gate) * up.
///
/// The activation is the sigmoid-weighted one, v / (1 + exp(-v)). It is NOT the
/// tanh-approximated gelu the other engine in the tree uses for its experts:
/// the two agree for positive arguments and differ by a tenth around v = -3,
/// which is exactly where expert gates sit, so trading one for the other is an
/// accuracy loss that no shape check ever notices.
__global__ void __launch_bounds__(kBlock)
    GateUpKernel(const float* gate, const float* up, float* act,
                 std::size_t groups, std::size_t count) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * kBlock + threadIdx.x;
  if (first < groups) {
    const float4 g = reinterpret_cast<const float4*>(gate)[first];
    const float4 u = reinterpret_cast<const float4*>(up)[first];
    float4 o;
    o.x = g.x / (1.0F + expf(-g.x)) * u.x;
    o.y = g.y / (1.0F + expf(-g.y)) * u.y;
    o.z = g.z / (1.0F + expf(-g.z)) * u.z;
    o.w = g.w / (1.0F + expf(-g.w)) * u.w;
    reinterpret_cast<float4*>(act)[first] = o;
  }
  const std::size_t tail = groups * kVec;
  if (first < count - tail) {
    const float g = gate[tail + first];
    act[tail + first] = g / (1.0F + expf(-g)) * up[tail + first];
  }
}

/// The end of a mixture block, per row r: the gated shared expert plus
/// the weight-weighted routed experts, with no normalization (the trunk adds
/// the raw mixture to the residual). One block per row; lanes split the row.
__global__ void __launch_bounds__(kBlock) MoeCombineKernel(MoeCombineArgs a) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= a.rows) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  const std::size_t first_slot = row * a.used;
  const float* weights = a.weights + first_slot;
  for (std::uint32_t i = lane; i < a.hidden; i += kBlock) {
    float m = a.shared[row * a.hidden + i];
    for (std::uint32_t e = 0; e < a.used; ++e) {
      m += weights[e] * a.routed[(first_slot + e) * a.hidden + i];
    }
    a.out[row * a.hidden + i] = m;
  }
}

void MoeCombine(const MoeCombineArgs& args, hipStream_t stream) {
  if (args.rows == 0) {
    return;
  }
  MoeCombineKernel<<<args.rows, kBlock, 0, stream>>>(args);
}

}  // namespace gufo::models::qwen35moe::rocm
