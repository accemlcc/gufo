#include "src/models/kolibri1/weights.hpp"

#include <cstdint>
#include <initializer_list>
#include <string>

namespace gufo::models::kolibri1 {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ6_K:
      return {256, 210};
    default:
      return {0, 0};
  }
}

/// Matrices the projections, the embeddings and the experts run in. The
/// released Q4_K_M artifact mixes the two K-quants per layer: attention and
/// the gate/up halves are Q4_K, while roughly a third of the `attn_v`,
/// `ffn_down_exps` and `ffn_down_shexp` tensors and the LM head keep Q6_K.
/// F32 is bound for the scalar oracle; a kernel dispatches on `TensorRef::type`
/// and rejects a format it cannot decode.
constexpr std::initializer_list<GgmlType> kMatrix = {
    GgmlType::kQ4_K, GgmlType::kQ6_K, GgmlType::kF32};
/// Norms, the router logits and the expert bias stay in F32 in every quant of
/// the artifact: routing decisions must not depend on a quantized bias.
constexpr std::initializer_list<GgmlType> kVector = {GgmlType::kF32};

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats;
  /// `experts` > 1 expects a stack of that many [cols, rows] matrices.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::initializer_list<GgmlType> types, bool required = true,
                std::uint64_t experts = 1) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly and
    // a payload running past its shard is caught here, not as a fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  /// Binds one block. Every layer has the same shape; only the per-layer
  /// sliding flag changes the attention path, not the tensors.
  LayerWeights Layer(const Config& c, std::uint32_t l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t q_dim = c.QDim();
    const std::uint64_t kv_dim = c.KvDim(l);
    LayerWeights w;
    w.attn_norm = Get(p + "attn_norm.weight", hidden, 1, kVector);
    w.attn_q = Get(p + "attn_q.weight", hidden, q_dim, kMatrix);
    w.attn_k = Get(p + "attn_k.weight", hidden, kv_dim, kMatrix);
    w.attn_v = Get(p + "attn_v.weight", hidden, kv_dim, kMatrix);
    w.attn_q_norm = Get(p + "attn_q_norm.weight", c.head_dim, 1, kVector);
    w.attn_k_norm = Get(p + "attn_k_norm.weight", c.head_dim, 1, kVector);
    w.attn_output = Get(p + "attn_output.weight", q_dim, hidden, kMatrix);
    w.post_attn_norm =
        Get(p + "post_attention_norm.weight", hidden, 1, kVector);
    w.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, kVector);

    w.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, kVector);
    w.router_bias = Get(p + "exp_probs_b.bias", c.num_experts, 1, kVector);
    w.gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ffn_size,
                      kMatrix, true, c.num_experts);
    w.up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ffn_size,
                    kMatrix, true, c.num_experts);
    w.down_exps = Get(p + "ffn_down_exps.weight", c.expert_ffn_size, hidden,
                      kMatrix, true, c.num_experts);
    w.gate_shexp = Get(p + "ffn_gate_shexp.weight", hidden,
                       c.shared_expert_ffn_size, kMatrix);
    w.up_shexp = Get(p + "ffn_up_shexp.weight", hidden,
                     c.shared_expert_ffn_size, kMatrix);
    w.down_shexp = Get(p + "ffn_down_shexp.weight", c.shared_expert_ffn_size,
                       hidden, kMatrix);
    w.post_ffn_norm = Get(p + "post_ffw_norm.weight", hidden, 1, kVector);
    return w;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  return f.block == 0 ? 0 : static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;

  // The vocabulary width is not a metadata key; it is the second dimension of
  // the embedding, which also sizes the LM head.
  const auto* embd = reader.FindTensor("token_embd.weight");
  if (embd == nullptr || embd->dimensions.size() != 2 ||
      embd->dimensions[0] != c.hidden_size || embd->dimensions[1] == 0) {
    b.Fail("token_embd.weight must be [hidden, vocab]");
    return std::nullopt;
  }
  const std::uint64_t vocab = embd->dimensions[1];
  w.token_embd = b.Get("token_embd.weight", c.hidden_size, vocab, kMatrix);
  // The released artifact ships an untied head. A tied one binds the same
  // view, so TiedOutput() stays exact rather than comparing names.
  w.output = b.Get("output.weight", c.hidden_size, vocab, kMatrix, false);
  if (b.ok && w.output.empty()) {
    w.output = w.token_embd;
  }
  w.output_norm = b.Get("output_norm.weight", c.hidden_size, 1, kVector);

  w.layers.reserve(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers && b.ok; ++l) {
    w.layers.push_back(b.Layer(c, l));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::kolibri1
