#ifndef GUFO_MODELS_KOLIBRI1_CONFIG_HPP_
#define GUFO_MODELS_KOLIBRI1_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::kolibri1 {

/// Routed experts per token and experts per layer the kernels support.
/// Kolibri 1 routes 6 of 384 in every one of its 50 layers.
inline constexpr std::uint32_t kMaxExpertsUsed = 8;
inline constexpr std::uint32_t kMaxExperts = 512;

/// Router rule carried by `expert_gating_func`. Kolibri 1 selects the top-k
/// experts on the biased raw logits (`logits + exp_probs_b`) and weights them
/// by the *unbiased* `sigmoid(logits)`, with no renormalisation. The
/// DeepSeek-V3 rule selects on `sigmoid(logits) + bias`, so it is a different
/// function even though both use a sigmoid and a bias.
enum class ExpertGating : std::uint8_t {
  kSigmoidLogitAdd = 5,  ///< GGUF `expert_gating_func` value 5.
};

/// Architecture parameters of a `kolibri1` target. Every value comes from the
/// artifact; the checks only bound what this runtime implements: text layers
/// in a sliding-window / global pattern, grouped-query attention with
/// per-head query and key norms, sandwich norms around both blocks, and
/// routed experts beside one ungated shared expert in every layer.
struct Config {
  std::uint32_t num_layers{0};      ///< 50
  std::uint32_t hidden_size{0};     ///< 2560
  std::uint32_t context_length{0};  ///< 262144
  float rms_eps{1e-6F};

  std::uint32_t num_heads{0};  ///< 48 query heads, the same per layer.
  std::vector<std::uint32_t> kv_heads;  ///< 4 per layer.
  std::vector<std::uint8_t> sliding;    ///< Per layer: 1 sliding, 0 global.

  std::uint32_t head_dim{0};       ///< 128 for both layer kinds.
  float rope_theta{0.0F};          ///< 1e4; the sliding layers only.
  std::uint32_t sliding_window{0}; ///< 513 keys, including the query.

  // Mixture of experts: `experts_used` routed experts plus
  // `shared_expert_count` ungated experts, in every layer.
  std::uint32_t num_experts{0};             ///< 384
  std::uint32_t experts_used{0};            ///< 6
  std::uint32_t expert_ffn_size{0};         ///< 512 per routed projection
  std::uint32_t shared_expert_count{0};     ///< 1
  std::uint32_t shared_expert_ffn_size{0};  ///< 512
  ExpertGating gating{ExpertGating::kSigmoidLogitAdd};
  bool renormalize_expert_weights{false};   ///< Kolibri routes without renorm.

  [[nodiscard]] bool IsSliding(std::uint32_t layer) const noexcept {
    return sliding[layer] != 0;
  }
  /// The global layers carry no positional encoding at all: RoPE runs in the
  /// sliding layers only.
  [[nodiscard]] bool UsesRope(std::uint32_t layer) const noexcept {
    return IsSliding(layer);
  }
  [[nodiscard]] std::uint32_t RopeDim(std::uint32_t layer) const noexcept {
    return UsesRope(layer) ? head_dim : 0;
  }
  [[nodiscard]] std::uint32_t QDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t KvDim(std::uint32_t layer) const noexcept {
    return kv_heads[layer] * head_dim;
  }
  [[nodiscard]] bool HasExperts() const noexcept { return num_experts != 0; }
  [[nodiscard]] std::uint32_t GlobalLayerCount() const noexcept;
  [[nodiscard]] std::uint32_t SlidingLayerCount() const noexcept {
    return num_layers - GlobalLayerCount();
  }

  /// Reads a `kolibri1` target. Rejects the variants this runtime does not
  /// implement: renormalised routing, a gating rule other than
  /// `sigmoid_logit_add`, no shared expert, or more than `kMaxExpertsUsed`
  /// experts per token out of more than `kMaxExperts` per layer.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_CONFIG_HPP_
