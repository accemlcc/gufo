#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_QKV_POST_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_QKV_POST_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen35moe::rocm {

/// Fused per-head post projection for a full-attention layer: the packed
/// Q|gate projection is deinterleaved per head ([q0, g0, q1, g1, ...] each
/// `dim` wide, unlike a flat halves split), Q and K are RMS-normalized and
/// rotated over the `rotary_dim` prefix, and K/V land in the layer's cache.
/// One wave per (row, head slot); slots are the query heads, then the key
/// heads, then the value heads.
struct QkvPostArgs {
  const float* qg;      ///< [rows][2 * heads * dim], packed Q and gate
  const float* k;       ///< [rows][kv_heads * dim]
  const float* v;       ///< [rows][kv_heads * dim]
  const float* q_norm;  ///< [dim], shared by every query head
  const float* k_norm;  ///< [dim], shared by every key head
  float* q_out;         ///< [rows][heads][dim], normed and rotated queries
  float* gate_out;      ///< [rows][heads * dim], the raw gate halves
  float* k_cache;       ///< [kv_heads][max_context][dim], F32
  float* v_cache;       ///< [kv_heads][max_context][dim], F32
  std::uint32_t rows;
  std::uint32_t first_position;  ///< cache slot of row 0
  std::uint32_t max_context;     ///< cache capacity in positions
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t dim;         ///< head dim (256)
  std::uint32_t rotary_dim;  ///< rotated prefix (64)
  double theta_scale;        ///< pow(rope_theta, -2 / rotary_dim)
  float eps;
};
void QkvPost(const QkvPostArgs& args, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_QKV_POST_HPP_
