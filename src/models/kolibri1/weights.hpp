#ifndef GUFO_MODELS_KOLIBRI1_WEIGHTS_HPP_
#define GUFO_MODELS_KOLIBRI1_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kolibri1/config.hpp"

namespace gufo::models::kolibri1 {

/// Non-owning view of one GGUF tensor. `cols` (ne[0]) is the contiguous
/// reduction dimension, `rows` (ne[1]) the output dimension and `experts`
/// (ne[2]) the number of stacked expert matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  /// Encoded bytes of one [cols, rows] matrix, all experts together.
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Distance between two stacked experts. It is a whole number of blocks, so
  /// one expert slice is itself a valid quant payload.
  [[nodiscard]] std::size_t ExpertBytes() const noexcept {
    return RowBytes() * rows;
  }
};

/// One of the 50 identical blocks. Every layer holds four sandwich norms: two
/// around attention and two around the mixture of experts. A pair normalises
/// the residual going into its branch and the branch output coming back.
struct LayerWeights {
  TensorRef attn_norm;      ///< [hidden]
  TensorRef attn_q;         ///< [hidden -> heads * head_dim]
  TensorRef attn_k;         ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;         ///< [hidden -> kv_heads * head_dim], every layer
  TensorRef attn_q_norm;    ///< [head_dim], shared by the query heads
  TensorRef attn_k_norm;    ///< [head_dim], shared by the KV heads
  TensorRef attn_output;    ///< [heads * head_dim -> hidden]
  TensorRef post_attn_norm;  ///< [hidden]
  TensorRef ffn_norm;        ///< [hidden], the pre-MoE norm

  // Routing: the top-k experts are selected on `logits + router_bias` and
  // weighted by the unbiased `sigmoid(logits)`, without renormalisation.
  TensorRef router;       ///< F32 [hidden -> experts]
  TensorRef router_bias;  ///< F32 [experts]

  // Routed experts, stacked along ne[2] in router index order, and the one
  // ungated shared expert that is added to the mixture with weight 1.
  TensorRef gate_exps;   ///< [experts][expert_ffn][hidden]
  TensorRef up_exps;     ///< [experts][expert_ffn][hidden]
  TensorRef down_exps;   ///< [experts][hidden][expert_ffn]
  TensorRef gate_shexp;  ///< [shared_ffn][hidden]
  TensorRef up_shexp;    ///< [shared_ffn][hidden]
  TensorRef down_shexp;  ///< [hidden][shared_ffn]

  TensorRef post_ffn_norm;  ///< [hidden]
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab]
  TensorRef output;       ///< LM head; the token embedding when tied
  TensorRef output_norm;  ///< [hidden]
  std::vector<LayerWeights> layers;

  [[nodiscard]] std::uint32_t VocabSize() const noexcept {
    return static_cast<std::uint32_t>(token_embd.rows);
  }
  [[nodiscard]] bool TiedOutput() const noexcept {
    return output.data == token_embd.data;
  }
  /// Binds and validates the target. Payloads stay mapped in the reader; only
  /// the headers and the small F32 vectors are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_WEIGHTS_HPP_
