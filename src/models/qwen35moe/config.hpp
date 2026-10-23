#ifndef GUFO_MODELS_QWEN35MOE_CONFIG_HPP_
#define GUFO_MODELS_QWEN35MOE_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen35moe {

/// Routed experts per token and experts per layer the kernels will support.
/// Ornith-1.5 routes 8 of 256 in every trunk layer, at 35B (A3B) and 122B.
inline constexpr std::uint32_t kMaxExpertsUsed = 16;
inline constexpr std::uint32_t kMaxExperts = 512;

/// Router rule of record: `Qwen3_5MoeTopKRouter` (transformers
/// `modeling_qwen3_5_moe.py`) applies softmax over the `ffn_gate_inp` logits,
/// keeps the top-k and renormalises the kept weights to sum 1. There is no
/// bias term and no `expert_gating_func` metadata; the shared expert is gated
/// by `sigmoid(ffn_gate_inp_shexp . x)`, not added at weight 1.
enum class ExpertGating : std::uint8_t {
  kSoftmaxTopKRenorm = 0,
};

/// Architecture parameters of a `qwen35moe` target. Every value comes from the
/// artifact; the checks only bound what this runtime will implement: a hybrid
/// stack of Gated DeltaNet (linear) and full-attention layers in a 3:1 ratio,
/// grouped-query attention with per-head-dim Q/K norms, mRoPE on the first
/// `rotary_dim` elements, zero-centered RMSNorms, and a softmax top-k mixture
/// with one sigmoid-gated shared expert in every trunk layer.
struct Config {
  std::uint32_t num_layers{0};      ///< Trunk layers, without the MTP block.
  std::uint32_t hidden_size{0};     ///< 2048 (35B) or 3072 (122B).
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  std::uint32_t num_heads{0};     ///< 16 (35B) or 32 (122B).
  std::uint32_t num_kv_heads{0};  ///< 2.
  std::uint32_t head_dim{0};      ///< 256.
  std::uint32_t rotary_dim{0};    ///< 64, the rotated prefix.
  std::array<std::uint32_t, 4> rope_sections{11, 11, 10, 0};
  float rope_theta{1e7F};

  std::uint32_t full_attention_interval{0};  ///< 4.
  /// Per trunk layer: 1 for Gated DeltaNet, 0 for full attention.
  std::vector<std::uint8_t> recurrent;

  // Gated DeltaNet (linear) layers.
  std::uint32_t ssm_conv_kernel{0};     ///< 4.
  std::uint32_t ssm_state_size{0};      ///< 128, both head dims.
  std::uint32_t ssm_group_count{0};     ///< 16 key heads.
  std::uint32_t ssm_time_step_rank{0};  ///< 64 value heads (35B: 32).
  std::uint32_t ssm_inner_size{0};      ///< 8192 (35B: 4096), the value dim.

  // Mixture of experts, in every trunk layer.
  std::uint32_t num_experts{0};      ///< 256.
  std::uint32_t experts_used{0};     ///< 8.
  std::uint32_t expert_ffn_size{0};  ///< 1024 (35B: 512).
  std::uint32_t shared_ffn_size{0};  ///< 1024 (35B: 512).
  ExpertGating gating{ExpertGating::kSoftmaxTopKRenorm};

  bool has_mtp{false};  ///< One `nextn` predictor block after the trunk.

  [[nodiscard]] bool IsLinear(std::uint32_t layer) const noexcept {
    return recurrent[layer] != 0;
  }
  /// Query/output width of a full-attention layer: q and gate packed.
  [[nodiscard]] std::uint32_t QProjDim() const noexcept {
    return 2 * num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t KvDim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttnOutDim() const noexcept {
    return num_heads * head_dim;
  }
  /// Key dim of the SSM projection: two key blocks plus the value block.
  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_group_count * ssm_state_size;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_inner_size;
  }
  [[nodiscard]] std::uint32_t SsmQkvDim() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t SsmValueHeads() const noexcept {
    return ssm_time_step_rank;
  }
  [[nodiscard]] std::uint32_t SsmValueHeadDim() const noexcept {
    return ssm_time_step_rank == 0 ? 0 : ssm_inner_size / ssm_time_step_rank;
  }
  [[nodiscard]] std::uint32_t LinearLayerCount() const noexcept;
  [[nodiscard]] std::uint32_t FullLayerCount() const noexcept {
    return num_layers - LinearLayerCount();
  }

  /// Reads a `qwen35moe` target. Rejects the variants this runtime does not
  /// implement: a missing or inconsistent layer pattern, SSM dims that do not
  /// factor, or an expert count outside the kernel bounds.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_CONFIG_HPP_
