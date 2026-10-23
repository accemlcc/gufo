#ifndef GUFO_MODELS_QWEN35MOE_REFERENCE_HPP_
#define GUFO_MODELS_QWEN35MOE_REFERENCE_HPP_

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen35moe/weights.hpp"

namespace gufo::models::qwen35moe {

/// Scalar reference of the qwen35moe graph, mirroring the transformers
/// implementation (`modeling_qwen3_5_moe.py`). Weights dequantize on the fly,
/// reductions accumulate in double, and every layer keeps full state for all
/// positions, so it only suits short prefixes. It pins the operator semantics
/// the kernels have to reproduce: standard RMSNorms (`y = rms(x) * w`; the
/// file stores the baked weights, so no +1 is added), Gated DeltaNet linear
/// recurrent state, full-attention layers with packed q|gate, per-head-dim Q/K
/// norms, mRoPE on the rotated prefix and a sigmoid output gate, and a softmax
/// top-k mixture with renormalised weights beside one sigmoid-gated shared
/// expert in every trunk layer.
class Reference {
public:
  explicit Reference(const ModelWeights& weights);

  /// Appends `tokens` (GGUF vocabulary ids) after the current position.
  /// `logits` receives [tokens.size()][vocab] rows; processing a prefix in
  /// several calls gives identical results.
  void Forward(std::span<const std::int32_t> tokens,
               std::vector<float>* logits);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  /// out[r][o] = sum_c W[o][c] * x[r][c] for `rows` input rows, W row-major
  /// [rows_out][cols] after dequantization.
  void MatMul(const void* data, core::GgmlType type, std::size_t rows_out,
              std::size_t cols, const float* x, std::size_t rows,
              float* out) const;
  void MatMulRef(const TensorRef& t, std::size_t rows_out, std::size_t cols,
                 const float* x, std::size_t rows, float* out) const {
    MatMul(t.data, t.type, rows_out, cols, x, rows, out);
  }
  /// One Gated DeltaNet layer over `rows` rows, updating the recurrent and
  /// convolution state.
  void LinearLayer(std::uint32_t layer, const float* x, std::size_t rows,
                   float* out);
  /// One full-attention layer over `rows` rows, updating the K/V cache.
  void FullLayer(std::uint32_t layer, const float* x, std::size_t rows,
                 float* out);
  /// Softmax top-k mixture plus the sigmoid-gated shared expert.
  void Mixture(const LayerWeights& w, const float* x, std::size_t rows,
               float* out) const;

  const ModelWeights& weights_;
  /// Per linear layer: recurrent state [v_heads][k_dim][v_dim] (float) and
  /// convolution window [qkv_dim][conv_kernel - 1].
  struct LinearState {
    std::vector<float> recurrent;
    std::vector<float> conv;
  };
  std::vector<LinearState> linear_;
  /// Per full layer: K/V after norm+RoPE, [position][kv_heads][head_dim].
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
  std::uint32_t position_{0};
};

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_REFERENCE_HPP_
