#ifndef GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_QKV_POST_HPP_
#define GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_QKV_POST_HPP_

#include <cstdint>
#include <hip/hip_runtime.h>

namespace gufo::models::kolibri1::rocm {

/// Per-head normalization of queries and keys, the rotation, and the cache
/// write: the step between the attention projections and the attention kernel.
///
/// Both norms are per head - one weight per dimension of a head, the same
/// weight for every head - which is why this is a step of its own and not a
/// norm over the projection. The rotation is NEOX: the pair (d, d + head_dim /
/// 2) of a head turns by position * theta^(-2d/head_dim), and it covers every
/// dimension; the released file carries no rope dimension count, so nothing is
/// left unturned.
///
/// `apply_rope` is false for the layers without positional encoding. Those
/// layers read the whole history anyway, and angles they never use are not
/// worth their cost.
///
/// Keys and values land in the cache as binary16 at slot (position % ring)
/// with all key heads of a position back to back - the layout the attention
/// kernel reads - and queries stay FP32 where they are.
struct QkvPostArgs {
  float* q;                ///< [rows][heads][head_dim], rewritten
  const float* k;          ///< [rows][kv_heads][head_dim]
  const float* v;          ///< [rows][kv_heads][head_dim]
  const float* q_norm;     ///< [head_dim]
  const float* k_norm;     ///< [head_dim]
  std::uint16_t* k_cache;  ///< binary16 [capacity][kv_heads][head_dim]
  std::uint16_t* v_cache;  ///< binary16 [capacity][kv_heads][head_dim]

  std::uint32_t rows;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;  ///< 128, the released geometry
  std::uint32_t first_position;
  std::uint32_t ring;  ///< cache capacity in positions; zero indexes by position
  float eps;
  /// theta^(-2/head_dim): pair d turns by position * theta_scale^d. It is a
  /// double, and the angles are computed in double, because a float angle at a
  /// 262144-position context has an ulp of 0.03 - and a cosine turns by that
  /// much, which is more than the whole attention score is worth.
  double theta_scale;
  bool apply_rope;
};
void QkvPost(const QkvPostArgs& args, hipStream_t stream);

}  // namespace gufo::models::kolibri1::rocm

#endif  // GUFO_MODELS_KOLIBRI1_KERNELS_ROCM_QKV_POST_HPP_
