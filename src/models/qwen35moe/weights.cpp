#include "src/models/qwen35moe/weights.hpp"

#include <cstdint>
#include <string_view>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen35moe {
namespace {

TensorRef ExtractTensorRef(const core::GgufReader& reader,
                           std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  TensorRef ref;
  ref.data = tensor->data;
  ref.type = tensor->type;
  ref.num_elements =
      static_cast<std::size_t>(tensor->ElementCount());
  const auto& d = tensor->dimensions;
  ref.cols = d.size() > 0 ? d[0] : 1;
  ref.rows = d.size() > 1 ? d[1] : 1;
  ref.experts = d.size() > 2 ? d[2] : 1;
  ref.name = tensor->name;
  // Locate the shard so the device upload can address the payload directly
  // and a payload running past its shard is caught here, not as a fault on
  // the device.
  const auto regions = reader.GetMappedRegions();
  const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
  for (std::uint32_t i = 0; i < regions.size(); ++i) {
    const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
    if (address >= base && address < base + regions[i].size) {
      ref.shard = i;
      ref.file_offset = address - base;
      break;
    }
  }
  return ref;
}

enum class TensorRole : std::uint8_t {
  kEmbedding,
  kNorm,
  kRouter,
  kProjection,
};

[[nodiscard]] constexpr std::string_view TensorRoleName(
    TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      return "embedding";
    case TensorRole::kNorm:
      return "normalization";
    case TensorRole::kRouter:
      return "router";
    case TensorRole::kProjection:
      return "projection";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool SupportsTensorType(core::GgmlType type,
                                               TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      // The UD-Q4_K_XL shard keeps the embedding at Q8_0 and the Q5_K_M file
      // at Q5_K; the lookup kernels decode any sub-16 format directly.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
             type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ5_K ||
             type == core::GgmlType::kQ6_K || type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
    case TensorRole::kNorm:
    case TensorRole::kRouter:
      // Norms, the router matrix, the shared-expert gate vector and the SSM
      // biases enter the arithmetic as floats. Accepting packed data would
      // reinterpret bytes.
      return type == core::GgmlType::kF32;
    case TensorRole::kProjection:
      // The released quants mix Q8_0, Q4_K, Q5_K, Q6_K and IQ4_XS across
      // projections, and every one of them has an in-kernel decoder on the
      // reuse targets (qwen dense, kolibri1 grouped GEMV).
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_K || type == core::GgmlType::kQ8_0 ||
             type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
             type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ3_K ||
             type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
  }
  return false;
}

bool ValidateTensor(const TensorRef& tensor, std::size_t expected_elements,
                    TensorRole role, std::string_view name,
                    std::string* error_msg) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required qwen35moe tensor: " + std::string(name);
    }
    return false;
  }
  if (!SupportsTensorType(tensor.type, role)) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported " + std::string(TensorRoleName(role)) +
                   " tensor type for " + std::string(name);
    }
    return false;
  }
  if (tensor.num_elements != expected_elements) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor shape mismatch for " + std::string(name) +
                   ": expected " + std::to_string(expected_elements) +
                   " elements, found " + std::to_string(tensor.num_elements);
    }
    return false;
  }
  return true;
}

bool BindFullAttention(const core::GgufReader& reader, const Config& config,
                       const std::string& prefix, LayerWeights* layer,
                       std::string* error_msg) {
  const std::size_t h = config.hidden_size;
  layer->is_full_attention = true;
  layer->attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
  layer->attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
  layer->attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
  layer->attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
  layer->attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
  layer->attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
  return ValidateTensor(layer->attn_q, h * config.QProjDim(),
                        TensorRole::kProjection, prefix + "attn_q.weight",
                        error_msg) &&
         ValidateTensor(layer->attn_k, h * config.KvDim(),
                        TensorRole::kProjection, prefix + "attn_k.weight",
                        error_msg) &&
         ValidateTensor(layer->attn_v, h * config.KvDim(),
                        TensorRole::kProjection, prefix + "attn_v.weight",
                        error_msg) &&
         ValidateTensor(layer->attn_output, config.AttnOutDim() * h,
                        TensorRole::kProjection, prefix + "attn_output.weight",
                        error_msg) &&
         ValidateTensor(layer->attn_q_norm, config.head_dim,
                        TensorRole::kNorm, prefix + "attn_q_norm.weight",
                        error_msg) &&
         ValidateTensor(layer->attn_k_norm, config.head_dim,
                        TensorRole::kNorm, prefix + "attn_k_norm.weight",
                        error_msg);
}

bool BindLinear(const core::GgufReader& reader, const Config& config,
                const std::string& prefix, LayerWeights* layer,
                std::string* error_msg) {
  const std::size_t h = config.hidden_size;
  layer->is_full_attention = false;
  layer->attn_qkv = ExtractTensorRef(reader, prefix + "attn_qkv.weight");
  layer->attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
  layer->ssm_a = ExtractTensorRef(reader, prefix + "ssm_a");
  layer->ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d.weight");
  layer->ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.bias");
  layer->ssm_alpha = ExtractTensorRef(reader, prefix + "ssm_alpha.weight");
  layer->ssm_beta = ExtractTensorRef(reader, prefix + "ssm_beta.weight");
  layer->ssm_norm = ExtractTensorRef(reader, prefix + "ssm_norm.weight");
  layer->ssm_out = ExtractTensorRef(reader, prefix + "ssm_out.weight");
  return ValidateTensor(layer->attn_qkv, h * config.SsmQkvDim(),
                        TensorRole::kProjection, prefix + "attn_qkv.weight",
                        error_msg) &&
         ValidateTensor(layer->attn_gate, h * config.SsmValueDim(),
                        TensorRole::kProjection, prefix + "attn_gate.weight",
                        error_msg) &&
         ValidateTensor(layer->ssm_a, config.SsmValueHeads(),
                        TensorRole::kNorm, prefix + "ssm_a", error_msg) &&
         ValidateTensor(
             layer->ssm_conv1d,
             static_cast<std::size_t>(config.ssm_conv_kernel) *
                 config.SsmQkvDim(),
             TensorRole::kNorm, prefix + "ssm_conv1d.weight", error_msg) &&
         ValidateTensor(layer->ssm_dt, config.SsmValueHeads(),
                        TensorRole::kNorm, prefix + "ssm_dt.bias", error_msg) &&
         ValidateTensor(layer->ssm_alpha, h * config.SsmValueHeads(),
                        TensorRole::kProjection, prefix + "ssm_alpha.weight",
                        error_msg) &&
         ValidateTensor(layer->ssm_beta, h * config.SsmValueHeads(),
                        TensorRole::kProjection, prefix + "ssm_beta.weight",
                        error_msg) &&
         ValidateTensor(layer->ssm_norm, config.SsmValueHeadDim(),
                        TensorRole::kNorm, prefix + "ssm_norm.weight",
                        error_msg) &&
         ValidateTensor(layer->ssm_out, config.SsmValueDim() * h,
                        TensorRole::kProjection, prefix + "ssm_out.weight",
                        error_msg);
}

bool BindMixture(const core::GgufReader& reader, const Config& config,
                 const std::string& prefix, LayerWeights* layer,
                 std::string* error_msg) {
  const std::size_t h = config.hidden_size;
  const std::size_t e = config.num_experts;
  const std::size_t f = config.expert_ffn_size;
  const std::size_t s = config.shared_ffn_size;
  layer->ffn_gate_inp =
      ExtractTensorRef(reader, prefix + "ffn_gate_inp.weight");
  layer->ffn_gate_inp_shexp =
      ExtractTensorRef(reader, prefix + "ffn_gate_inp_shexp.weight");
  layer->ffn_gate_shexp =
      ExtractTensorRef(reader, prefix + "ffn_gate_shexp.weight");
  layer->ffn_up_shexp =
      ExtractTensorRef(reader, prefix + "ffn_up_shexp.weight");
  layer->ffn_down_shexp =
      ExtractTensorRef(reader, prefix + "ffn_down_shexp.weight");
  layer->ffn_gate_exps =
      ExtractTensorRef(reader, prefix + "ffn_gate_exps.weight");
  layer->ffn_up_exps = ExtractTensorRef(reader, prefix + "ffn_up_exps.weight");
  layer->ffn_down_exps =
      ExtractTensorRef(reader, prefix + "ffn_down_exps.weight");
  return ValidateTensor(layer->ffn_gate_inp, h * e, TensorRole::kRouter,
                        prefix + "ffn_gate_inp.weight", error_msg) &&
         ValidateTensor(layer->ffn_gate_inp_shexp, h, TensorRole::kRouter,
                        prefix + "ffn_gate_inp_shexp.weight", error_msg) &&
         ValidateTensor(layer->ffn_gate_shexp, h * s, TensorRole::kProjection,
                        prefix + "ffn_gate_shexp.weight", error_msg) &&
         ValidateTensor(layer->ffn_up_shexp, h * s, TensorRole::kProjection,
                        prefix + "ffn_up_shexp.weight", error_msg) &&
         ValidateTensor(layer->ffn_down_shexp, s * h, TensorRole::kProjection,
                        prefix + "ffn_down_shexp.weight", error_msg) &&
         ValidateTensor(layer->ffn_gate_exps, h * f * e,
                        TensorRole::kProjection,
                        prefix + "ffn_gate_exps.weight", error_msg) &&
         ValidateTensor(layer->ffn_up_exps, h * f * e,
                        TensorRole::kProjection,
                        prefix + "ffn_up_exps.weight", error_msg) &&
         ValidateTensor(layer->ffn_down_exps, f * h * e,
                        TensorRole::kProjection,
                        prefix + "ffn_down_exps.weight", error_msg);
}

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  return quant::EncodedSizeBytes(type, static_cast<std::size_t>(cols));
}

std::optional<ModelWeights> ModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto config_opt = Config::FromGguf(reader, error_msg);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }

  ModelWeights weights;
  weights.config = *config_opt;
  const Config& config = weights.config;
  const std::size_t h = config.hidden_size;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    weights.output = weights.token_embd;
  }
  // The vocabulary falls out of the embedding rows; both released files
  // carry 248320 entries, the Qwen3.5/3.8 vocabulary.
  std::uint64_t vocab = 0;
  if (!weights.token_embd.empty() && h != 0) {
    vocab = weights.token_embd.num_elements / h;
  }
  if (!ValidateTensor(weights.token_embd, vocab * h, TensorRole::kEmbedding,
                      "token_embd.weight", error_msg) ||
      vocab == 0 ||
      !ValidateTensor(weights.output_norm, h, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, vocab * h, TensorRole::kProjection,
                      "output.weight", error_msg)) {
    return std::nullopt;
  }

  weights.layers.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    // The tensor names are the ground truth of the layer kind: a converter
    // that names a layer differently than the metadata pattern fails here,
    // with the layer numbered, instead of binding the wrong graph.
    const bool has_q = reader.HasTensor(prefix + "attn_q.weight");
    const bool has_qkv = reader.HasTensor(prefix + "attn_qkv.weight");
    if (has_q == has_qkv) {
      if (error_msg != nullptr) {
        *error_msg = "Layer blk." + std::to_string(i) +
                     " has an ambiguous attention tensor set";
      }
      return std::nullopt;
    }
    if (has_q == config.IsLinear(i)) {
      if (error_msg != nullptr) {
        *error_msg = "Layer blk." + std::to_string(i) +
                     " disagrees with the recurrent pattern";
      }
      return std::nullopt;
    }
    layer.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    layer.post_attention_norm =
        ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    if (!ValidateTensor(layer.attn_norm, h, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(layer.post_attention_norm, h, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg)) {
      return std::nullopt;
    }
    const bool want_full = !config.IsLinear(i);
    const bool ok = want_full
                        ? BindFullAttention(reader, config, prefix, &layer,
                                            error_msg)
                        : BindLinear(reader, config, prefix, &layer, error_msg);
    if (!ok) {
      return std::nullopt;
    }
    if (!BindMixture(reader, config, prefix, &layer, error_msg)) {
      return std::nullopt;
    }
  }

  if (config.has_mtp) {
    const std::string prefix =
        "blk." + std::to_string(config.num_layers) + ".";
    if (!reader.HasTensor(prefix + "attn_q.weight") ||
        reader.HasTensor(prefix + "attn_qkv.weight")) {
      if (error_msg != nullptr) {
        *error_msg = "Layer " + prefix +
                     " (nextn predictor) is not a full-attention layer";
      }
      return std::nullopt;
    }
    auto& mtp = weights.mtp.block;
    mtp.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    mtp.post_attention_norm =
        ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    if (!ValidateTensor(mtp.attn_norm, h, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(mtp.post_attention_norm, h, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg) ||
        !BindFullAttention(reader, config, prefix, &mtp, error_msg) ||
        !BindMixture(reader, config, prefix, &mtp, error_msg)) {
      return std::nullopt;
    }
    weights.mtp.eh_proj =
        ExtractTensorRef(reader, prefix + "nextn.eh_proj.weight");
    weights.mtp.enorm =
        ExtractTensorRef(reader, prefix + "nextn.enorm.weight");
    weights.mtp.hnorm =
        ExtractTensorRef(reader, prefix + "nextn.hnorm.weight");
    weights.mtp.shared_head_norm =
        ExtractTensorRef(reader, prefix + "nextn.shared_head_norm.weight");
    if (!ValidateTensor(weights.mtp.eh_proj, 2 * h * h,
                        TensorRole::kProjection,
                        prefix + "nextn.eh_proj.weight", error_msg) ||
        !ValidateTensor(weights.mtp.enorm, h, TensorRole::kNorm,
                        prefix + "nextn.enorm.weight", error_msg) ||
        !ValidateTensor(weights.mtp.hnorm, h, TensorRole::kNorm,
                        prefix + "nextn.hnorm.weight", error_msg) ||
        !ValidateTensor(weights.mtp.shared_head_norm, h, TensorRole::kNorm,
                        prefix + "nextn.shared_head_norm.weight", error_msg)) {
      return std::nullopt;
    }
  }
  return weights;
}

}  // namespace gufo::models::qwen35moe
