#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_SSM_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_SSM_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen35moe::rocm {

/// Gated DeltaNet token mixer over `rows` rows: the depthwise causal
/// convolution with SiLU, then the recurrent gated-delta update with per-head
/// RMSNorm and SiLU gating. The K/V heads tile across the value heads
/// (`h % num_key_heads`), matching the checked-in file layout and ggml.
///
/// State is the caller's: `conv_state` holds four past inputs per channel and
/// `recurrent` the [value_heads][state][head_dim] matrix, both F32 and both
/// preserved across forwards.
struct SsmArgs {
  const float* qkv;       ///< [rows][qkv_dim], the joint projection
  const float* conv_w;    ///< F32 [qkv_dim][conv_kernel]
  float* conv_state;      ///< [qkv_dim * conv_kernel], persistent
  float* conv_out;        ///< [rows][qkv_dim]
  float* recurrent;       ///< [value_heads][state][head_dim], persistent
  const float* alpha;     ///< [rows][value_heads]
  const float* beta;      ///< [rows][value_heads]
  const float* ssm_a;     ///< F32 [value_heads], negative decay rates
  const float* ssm_dt;    ///< F32 [value_heads]
  const float* ssm_norm;  ///< F32 [head_dim], shared by every value head
  const float* gate;      ///< [rows][value_dim], the output gate z
  float* out;             ///< [rows][value_dim], normed and gated
  std::uint32_t rows;
  std::uint32_t qkv_dim;  ///< 2 * key_dim + value_dim
  std::uint32_t num_key_heads;
  std::uint32_t num_heads;  ///< value heads
  std::uint32_t key_dim;    ///< 128
  std::uint32_t value_dim;  ///< value heads * head_dim
  std::uint32_t head_dim;   ///< 128
};
void LaunchSsm(const SsmArgs& args, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_SSM_HPP_
