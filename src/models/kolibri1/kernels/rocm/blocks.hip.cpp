#include <cstdint>
#include <stdexcept>

#include "src/models/kolibri1/kernels/rocm/moe.hpp"

namespace gufo::models::kolibri1::rocm {
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

/// The end of a mixture block. `kLanes` threads take one output row, so the
/// shared row and the `used` expert rows belonging to it are read side by side
/// and each of those reads is coalesced; a thread per row would walk the seven
/// rows of a token one element at a time.
///
/// The contributions add in ascending expert index and the shared one is folded
/// in first, so a row cannot pick up a neighbour's rounding and a change of
/// batch leaves its value alone.
template<int kLanes, int kWindow>
__global__ void __launch_bounds__(kBlock) MoeFinishKernel(MoeFinishArgs a) {
  constexpr int kRowsPerBlock = kBlock / kLanes;
  [[maybe_unused]] __shared__ float
      scratch[kLanes > kWave ? kLanes / kWave : 1];
  const std::uint32_t lane = threadIdx.x % kLanes;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * kRowsPerBlock +
                          threadIdx.x / kLanes;
  if (kLanes == kWave && row >= a.rows) {
    return;
  }
  const std::uint32_t groups = Groups(a.hidden);
  const std::size_t first_slot = row * a.used;
  const float4* shared_row = reinterpret_cast<const float4*>(a.shared) +
                             static_cast<std::size_t>(row) * groups;
  const float4* routed = reinterpret_cast<const float4*>(a.routed);
  const float* weights = a.weights + first_slot;
  float4 mix[kWindow];
  float ss = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kWindow; ++j) {
    const std::uint32_t g = lane + j * kLanes;
    if (g < groups) {
      float4 m = shared_row[g];
      for (std::uint32_t e = 0; e < a.used; ++e) {
        const float4 v = routed[(first_slot + e) * groups + g];
        const float w = weights[e];
        m.x += w * v.x;
        m.y += w * v.y;
        m.z += w * v.z;
        m.w += w * v.w;
      }
      mix[j] = m;
      ss += m.x * m.x;
      ss += m.y * m.y;
      ss += m.z * m.z;
      ss += m.w * m.w;
    } else {
      mix[j] = Zero4();
    }
  }
  const float mean = RowSum<kLanes>(ss, scratch) / static_cast<float>(a.hidden);
  const float r = 1.0F / sqrtf(mean + a.eps);
  float4* xr =
      reinterpret_cast<float4*>(a.x) + static_cast<std::size_t>(row) * groups;
  const float4* nr = reinterpret_cast<const float4*>(a.post_norm);
#pragma unroll
  for (std::uint32_t j = 0; j < kWindow; ++j) {
    const std::uint32_t g = lane + j * kLanes;
    if (g >= groups) {
      break;
    }
    const float4 x = xr[g];
    const float4 n = nr[g];
    const float4 m = mix[j];
    xr[g] = {x.x + m.x * r * n.x, x.y + m.y * r * n.y, x.z + m.z * r * n.z,
             x.w + m.w * r * n.w};
  }
}

/// Launches the norm over the grid with one row per `kLanes` threads.
template<int kLanes, int kWindow>
void LaunchNormAt(const NormRowsArgs& a, hipStream_t stream) {
  const std::uint32_t rows_per_block = kBlock / kLanes;
  const dim3 grid((a.rows + rows_per_block - 1) / rows_per_block);
  RmsNormKernel<kLanes, kWindow><<<grid, kBlock, 0, stream>>>(a);
}

/// The narrowest register window that still covers the row. `n` picks it and
/// nothing else does, which is the whole reason a row sums the same way in a
/// one-token decode as in a prefill.
template<int kLanes, int kWindow>
void LaunchNorm(const NormRowsArgs& a, hipStream_t stream) {
  if constexpr (kWindow <
                (kLanes == kWave ? kMaxWaveWindow : kMaxBlockWindow)) {
    if (Groups(a.n) <= static_cast<std::uint32_t>(kLanes) * kWindow) {
      LaunchNormAt<kLanes, kWindow>(a, stream);
      return;
    }
    LaunchNorm<kLanes, kWindow + 1>(a, stream);
  } else {
    LaunchNormAt<kLanes, kWindow>(a, stream);
  }
}

template<int kLanes, int kWindow>
void LaunchFinishAt(const MoeFinishArgs& a, hipStream_t stream) {
  const std::uint32_t rows_per_block = kBlock / kLanes;
  const dim3 grid((a.rows + rows_per_block - 1) / rows_per_block);
  MoeFinishKernel<kLanes, kWindow><<<grid, kBlock, 0, stream>>>(a);
}

template<int kLanes, int kWindow>
void LaunchFinish(const MoeFinishArgs& a, hipStream_t stream) {
  if constexpr (kWindow <
                (kLanes == kWave ? kMaxWaveWindow : kMaxBlockWindow)) {
    if (Groups(a.hidden) <= static_cast<std::uint32_t>(kLanes) * kWindow) {
      LaunchFinishAt<kLanes, kWindow>(a, stream);
      return;
    }
    LaunchFinish<kLanes, kWindow + 1>(a, stream);
  } else {
    LaunchFinishAt<kLanes, kWindow>(a, stream);
  }
}

/// Enough blocks for the whole vectors of a flat count, with one spare so a
/// count that ends mid-vector still has threads left for its tail.
[[nodiscard]] unsigned ElementBlocks(std::size_t groups) noexcept {
  return static_cast<unsigned>(groups / kBlock + 1);
}

/// The vector bodies need a 16-byte aligned base. Rows inherit that from a
/// length that is a multiple of 32, so this is only ever about the pointer the
/// caller handed over.
[[nodiscard]] bool Aligned(const void* p) noexcept {
  return reinterpret_cast<std::uintptr_t>(p) % (kVec * sizeof(float)) == 0;
}

}  // namespace

void RmsNormRows(const NormRowsArgs& args, hipStream_t stream) {
  if (args.n % 32 != 0) {
    throw std::invalid_argument(
        "RmsNormRows needs a row length that is a multiple of 32");
  }
  if (Groups(args.n) > static_cast<std::uint32_t>(kBlock) * kMaxBlockWindow) {
    throw std::invalid_argument("RmsNormRows supports rows up to 8192 floats");
  }
  if (args.rows == 0) {
    return;
  }
  if (Groups(args.n) <= static_cast<std::uint32_t>(kWave) * kMaxWaveWindow) {
    LaunchNorm<kWave, 1>(args, stream);
  } else {
    LaunchNorm<kBlock, 1>(args, stream);
  }
}

void AddRows(const AddRowsArgs& args, hipStream_t stream) {
  if (args.count == 0) {
    return;
  }
  if (!Aligned(args.x) || !Aligned(args.add)) {
    throw std::invalid_argument("AddRows needs 16-byte aligned buffers");
  }
  AddKernel<<<ElementBlocks(args.count / kVec), kBlock, 0, stream>>>(
      args.x, args.add, args.count / kVec, args.count);
}

void GateUpAct(const GateUpArgs& args, hipStream_t stream) {
  if (args.count == 0) {
    return;
  }
  if (!Aligned(args.gate) || !Aligned(args.up) || !Aligned(args.act)) {
    throw std::invalid_argument("GateUpAct needs 16-byte aligned buffers");
  }
  GateUpKernel<<<ElementBlocks(args.count / kVec), kBlock, 0, stream>>>(
      args.gate, args.up, args.act, args.count / kVec, args.count);
}

void MoeFinish(const MoeFinishArgs& args, hipStream_t stream) {
  if (args.hidden % 32 != 0) {
    throw std::invalid_argument(
        "MoeFinish needs a hidden size that is a multiple of 32");
  }
  if (Groups(args.hidden) >
      static_cast<std::uint32_t>(kBlock) * kMaxBlockWindow) {
    throw std::invalid_argument("MoeFinish supports hidden up to 8192");
  }
  if (args.rows == 0) {
    return;
  }
  if (Groups(args.hidden) <=
      static_cast<std::uint32_t>(kWave) * kMaxWaveWindow) {
    LaunchFinish<kWave, 1>(args, stream);
  } else {
    LaunchFinish<kBlock, 1>(args, stream);
  }
}

}  // namespace gufo::models::kolibri1::rocm
