#ifndef GUFO_MODELS_QWEN35MOE_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN35MOE_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35moe/config.hpp"

namespace gufo::models::qwen35moe {

/// Borrowed view of one mapped tensor. Weights are never copied here; the
/// device upload reads them in place from the mapped shards.
struct TensorRef {
  const void* data = nullptr;
  core::GgmlType type = core::GgmlType::kF32;
  std::size_t num_elements = 0;

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || num_elements == 0;
  }
};

/// Tensor references of one trunk layer. Linear layers carry the Gated
/// DeltaNet set, full-attention layers the GQA set; every trunk layer carries
/// the same mixture: a softmax router, one sigmoid-gated shared expert and
/// the routed experts. The MTP predictor block reuses the full-attention set
/// plus the `nextn.*` tensors below.
struct LayerWeights {
  bool is_full_attention = false;

  TensorRef attn_norm;

  // Full attention (every `full_attention_interval`-th trunk layer).
  TensorRef attn_q;  ///< [hidden, 2 * heads * head_dim]: query, then gate.
  TensorRef attn_k;  ///< [hidden, kv_heads * head_dim].
  TensorRef attn_v;  ///< [hidden, kv_heads * head_dim].
  TensorRef attn_output;  ///< [heads * head_dim, hidden].
  TensorRef attn_q_norm;  ///< [head_dim], zero-centered RMSNorm.
  TensorRef attn_k_norm;  ///< [head_dim], zero-centered RMSNorm.

  // Gated DeltaNet (all other trunk layers).
  TensorRef attn_qkv;    ///< [hidden, 2 * key_dim + value_dim].
  TensorRef attn_gate;   ///< [hidden, value_dim], the output gate z.
  TensorRef ssm_a;       ///< [value_heads], log decay rates.
  TensorRef ssm_conv1d;  ///< [conv_kernel, qkv_dim] depthwise.
  TensorRef ssm_dt;      ///< [value_heads], discretization bias.
  TensorRef ssm_alpha;   ///< [hidden, value_heads], time-step branch.
  TensorRef ssm_beta;    ///< [hidden, value_heads], rate branch.
  TensorRef ssm_norm;    ///< [value_head_dim], gated RMSNorm.
  TensorRef ssm_out;     ///< [value_dim, hidden].

  // Mixture, in every trunk layer.
  TensorRef ffn_gate_inp;  ///< [hidden, experts] F32 router matrix.
  TensorRef ffn_gate_inp_shexp;  ///< [hidden] F32 shared-expert gate vector.
  TensorRef ffn_gate_shexp;      ///< [hidden, shared_ff].
  TensorRef ffn_up_shexp;        ///< [hidden, shared_ff].
  TensorRef ffn_down_shexp;      ///< [shared_ff, hidden].
  TensorRef ffn_gate_exps;       ///< [hidden, expert_ff, experts].
  TensorRef ffn_up_exps;         ///< [hidden, expert_ff, experts].
  TensorRef ffn_down_exps;       ///< [expert_ff, hidden, experts].

  TensorRef post_attention_norm;
};

/// The `nextn` predictor block: full-attention tensors plus its own norms and
/// the embedding/hidden projection. Bound but not executed until the draft
/// path lands; a file without it fails here rather than later.
struct MtpWeights {
  LayerWeights block;
  TensorRef eh_proj;  ///< [2 * hidden, hidden].
  TensorRef enorm;    ///< [hidden].
  TensorRef hnorm;    ///< [hidden].
  TensorRef shared_head_norm;  ///< [hidden].
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;
  std::vector<LayerWeights> layers;
  TensorRef output_norm;
  TensorRef output;  ///< Falls back to `token_embd` on tied files.
  MtpWeights mtp;    ///< Valid only when `config.has_mtp` is set.

  /// Binds every tensor by name and checks shapes and formats. No weight is
  /// copied; an unexpected per-tensor layout fails here with the tensor
  /// named instead of decoding wrong numbers on the device.
  [[nodiscard]] static std::optional<ModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_WEIGHTS_HPP_
