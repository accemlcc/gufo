#ifndef GUFO_MODELS_KOLIBRI1_REFERENCE_HPP_
#define GUFO_MODELS_KOLIBRI1_REFERENCE_HPP_

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "src/models/kolibri1/weights.hpp"

namespace gufo::models::kolibri1 {

/// Scalar reference of the Kolibri 1 graph, mirroring the model definition of
/// the Hob-forge conversion (`llama_model_kolibri1`). Weights dequantize on
/// the fly, every reduction accumulates in double, and each layer keeps full
/// K/V for all positions, so it only suits short prefixes. It pins the
/// operator semantics the kernels have to reproduce: per-head query and key
/// RMS norms whose weight every head of a kind shares, RoPE in the sliding
/// layers and none at all in the global ones, a 513 key window against full
/// causal attention, the score scale 1/sqrt(head_dim), the sandwich norms on
/// the branch outputs rather than on the residual, SwiGLU experts selected by
/// `top_k(logits + bias)` but weighted by the unbiased `sigmoid(logits)` with
/// no renormalisation, and the one ungated shared expert added with weight 1.
class Reference {
public:
  /// K/V stays in double and activations are never rounded, so a difference
  /// against the device path is the operators and the cache format, not the
  /// arithmetic the kernels cannot avoid.
  explicit Reference(const ModelWeights& weights);

  /// Appends `tokens` (GGUF vocabulary ids) after the current position.
  /// `logits` receives [tokens.size()][vocab] rows and `hidden` the post-norm
  /// rows fed to the LM head; either may be null. Processing a prefix in
  /// several calls gives identical results.
  void Forward(std::span<const std::int32_t> tokens, std::vector<float>* logits,
               std::vector<float>* hidden = nullptr);

  /// When enabled, Forward records the residual stream after every layer:
  /// [layer][row][hidden] for the rows of the latest call.
  void SetTrace(bool enabled) noexcept { trace_ = enabled; }
  [[nodiscard]] const std::vector<float>& LayerTrace() const noexcept {
    return layer_trace_;
  }

  /// Additionally records named intermediates under the names the reference
  /// graph gives its callbacks: `attn_norm-<l>`, `Qcur_normed-<l>`,
  /// `Qcur_rope-<l>`, `Kcur_normed-<l>`, `Kcur_rope-<l>`, `attn_out-<l>`,
  /// `attn_post_norm-<l>`, `ffn_inp-<l>`, `ffn_norm-<l>`, `ffn_moe_out-<l>`,
  /// `ffn_shexp-<l>`, `ffn_out-<l>`, `ffn_post_norm-<l>`, `l_out-<l>` and
  /// `result_norm`. Each blob is row-major with the innermost axis matching
  /// ggml's fastest axis, so a row of a llama.cpp tensor and a row here have
  /// the same element order and one operator can be compared on its own.
  void SetStageTrace(bool enabled) noexcept { stage_trace_enabled_ = enabled; }
  [[nodiscard]] const std::map<std::string, std::vector<float>>& StageTrace()
      const noexcept {
    return stage_trace_;
  }

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  /// out[r][o] = sum_c W[o][c] * x[r][c] for `rows` input rows.
  void MatMul(const TensorRef& weight, const float* x, std::size_t rows,
              float* out) const;
  /// Attention over the layer's cache, which already holds the new rows.
  void Attention(std::uint32_t layer, const float* q, std::size_t rows,
                 float* out) const;
  /// Routed experts into `routed` and the one shared expert into `shared`,
  /// separately because the reference graph traces them separately. Either may
  /// be null.
  void Mixture(const LayerWeights& w, const float* x, std::size_t rows,
               float* routed, float* shared) const;
  /// Stores one named intermediate when stage tracing is on.
  void Stage(const std::string& name, const float* data, std::size_t count);

  const ModelWeights& weights_;
  /// Per layer: [position][kv_heads][head_dim], K stored after RoPE.
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
  std::vector<float> layer_trace_;
  std::map<std::string, std::vector<float>> stage_trace_;
  std::uint32_t position_{0};
  bool trace_{false};
  bool stage_trace_enabled_{false};
};

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_REFERENCE_HPP_
