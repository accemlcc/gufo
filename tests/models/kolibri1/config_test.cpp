#include "src/models/kolibri1/config.hpp"
#include "src/models/kolibri1/weights.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// The in-memory GGUF writer and the always-on assertion live in this directory.
// They came out of the Gemma 4 host checks; copying them here is the cheaper
// trade, because a model's tests then fail for reasons inside that model.
#include "tests/models/kolibri1/check.hpp"
#include "tests/models/kolibri1/gguf_builder.hpp"

namespace k1 = gufo::models::kolibri1;
using kolibri1_test::BoolArray;
using kolibri1_test::GgufImage;
using kolibri1_test::I32Array;
using kolibri1_test::Metadata;
using kolibri1_test::Require;
using kolibri1_test::TensorSpec;
using kolibri1_test::Tensors;
using gufo::core::GgmlType;

namespace {

/// The pattern of the released artifact: four sliding layers per global layer,
/// counting from one, so a 50-layer model ends on a global layer.
std::vector<bool> Pattern(std::uint32_t layers) {
  std::vector<bool> pattern;
  pattern.reserve(layers);
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    pattern.push_back((layer + 1) % 5 != 0);
  }
  return pattern;
}

/// A two-layer miniature of Kolibri 1: one sliding and one global layer, with
/// the scalar KV head count and the routing metadata of the real file.
Metadata Base() {
  const std::string p = "kolibri1.";
  return {
      {"general.architecture", std::string("kolibri1")},
      {p + "block_count", std::uint32_t{2}},
      {p + "context_length", std::uint32_t{4096}},
      {p + "embedding_length", std::uint32_t{64}},
      {p + "attention.head_count", std::uint32_t{4}},
      {p + "attention.head_count_kv", std::uint32_t{2}},
      {p + "attention.sliding_window_pattern", BoolArray{{true, false}}},
      {p + "attention.key_length", std::uint32_t{128}},
      {p + "attention.value_length", std::uint32_t{128}},
      {p + "attention.sliding_window", std::uint32_t{16}},
      {p + "attention.layer_norm_rms_epsilon", 1e-6F},
      {p + "rope.freq_base", 10000.0F},
      {p + "expert_count", std::uint32_t{8}},
      {p + "expert_used_count", std::uint32_t{2}},
      {p + "expert_feed_forward_length", std::uint32_t{32}},
      {p + "expert_shared_count", std::uint32_t{1}},
      {p + "expert_shared_feed_forward_length", std::uint32_t{32}},
      {p + "expert_weights_norm", false},
      {p + "expert_gating_func", std::uint32_t{5}},
  };
}

/// The same miniature with widths that are multiples of the 256-element
/// K-quant block, so the Q4_K/Q6_K mix of the released artifact is legal.
Metadata QuantMetadata() {
  Metadata m = Base();
  const std::string p = "kolibri1.";
  m[p + "embedding_length"] = std::uint32_t{256};
  m[p + "expert_feed_forward_length"] = std::uint32_t{256};
  m[p + "expert_shared_feed_forward_length"] = std::uint32_t{256};
  return m;
}

std::optional<k1::Config> Parse(const Metadata& metadata,
                                std::string* error) {
  GgufImage image(metadata);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  auto config = k1::Config::FromGguf(*image.reader(), error);
  Require(config.has_value() == error->empty(),
          "error message must accompany every rejection");
  return config;
}

void ExpectRejected(Metadata metadata, const std::string& what) {
  std::string error;
  Require(!Parse(metadata, &error).has_value(), what + " was accepted");
  Require(!error.empty(), what + " was rejected without a message");
}

void CheckTarget() {
  std::string error;
  const auto c = Parse(Base(), &error);
  Require(c.has_value(), "valid kolibri1 metadata rejected: " + error);
  Require(c->num_layers == 2 && c->hidden_size == 64 &&
          c->context_length == 4096,
          "dimensions");
  Require(c->num_heads == 4 && c->kv_heads.size() == 2 && c->kv_heads[0] == 2,
          "the scalar KV head count must expand to every layer");
  Require(c->IsSliding(0) && !c->IsSliding(1), "sliding pattern");
  // RoPE runs where the window runs; the global layer is un-roped.
  Require(c->UsesRope(0) && !c->UsesRope(1), "positional pattern");
  Require(c->head_dim == 128 && c->RopeDim(0) == 128 && c->RopeDim(1) == 0,
          "rope dimensions");
  Require(c->QDim() == 4 * 128 && c->KvDim(0) == 2 * 128, "projection widths");
  Require(c->sliding_window == 16 && c->rope_theta == 10000.0F,
          "window and rope base");
  Require(c->GlobalLayerCount() == 1 && c->SlidingLayerCount() == 1,
          "layer counts");
  Require(c->num_experts == 8 && c->experts_used == 2 &&
              c->expert_ffn_size == 32,
          "routed expert geometry");
  Require(c->shared_expert_count == 1 && c->shared_expert_ffn_size == 32,
          "shared expert geometry");
  Require(c->gating == k1::ExpertGating::kSigmoidLogitAdd &&
              !c->renormalize_expert_weights,
          "routing rule");
}

/// The geometry of the 78B artifact: 50 layers in a 4:1 pattern, 48 query and
/// 4 KV heads of 128, a 513 key window, and 6 of 384 routed experts.
void CheckArtifactGeometry() {
  Metadata metadata = Base();
  const std::string p = "kolibri1.";
  metadata[p + "block_count"] = std::uint32_t{50};
  metadata[p + "context_length"] = std::uint32_t{262144};
  metadata[p + "embedding_length"] = std::uint32_t{2560};
  metadata[p + "attention.head_count"] = std::uint32_t{48};
  metadata[p + "attention.head_count_kv"] = std::uint32_t{4};
  metadata[p + "attention.sliding_window_pattern"] = BoolArray{Pattern(50)};
  metadata[p + "attention.sliding_window"] = std::uint32_t{513};
  metadata[p + "expert_count"] = std::uint32_t{384};
  metadata[p + "expert_used_count"] = std::uint32_t{6};
  metadata[p + "expert_feed_forward_length"] = std::uint32_t{512};
  metadata[p + "expert_shared_feed_forward_length"] = std::uint32_t{512};

  std::string error;
  const auto c = Parse(metadata, &error);
  Require(c.has_value(), "released kolibri1 geometry rejected: " + error);
  Require(c->SlidingLayerCount() == 40 && c->GlobalLayerCount() == 10,
          "the 4:1 pattern must yield 40 sliding and 10 global layers");
  Require(c->IsSliding(48) && !c->IsSliding(44) && !c->IsSliding(49),
          "the pattern must count from one, ending on a global layer");
  Require(c->QDim() == 48 * 128 && c->KvDim(0) == 4 * 128,
          "6144 query and 512 KV columns");
  Require(c->num_experts <= k1::kMaxExperts &&
              c->experts_used <= k1::kMaxExpertsUsed,
          "the artifact must fit the kernel bounds");
}

void CheckPerLayerValues() {
  Metadata per_layer = Base();
  const std::string p = "kolibri1.";
  per_layer[p + "attention.head_count_kv"] = I32Array{{2, 1}};
  std::string error;
  const auto c = Parse(per_layer, &error);
  Require(c.has_value(), "per-layer KV heads rejected: " + error);
  Require(c->kv_heads.size() == 2 && c->kv_heads[0] == 2 && c->kv_heads[1] == 1,
          "per-layer KV heads");
  Require(c->KvDim(0) == 2 * 128 && c->KvDim(1) == 1 * 128,
          "per-layer KV widths");

  Metadata short_kv = Base();
  short_kv[p + "attention.head_count_kv"] = I32Array{{2}};
  ExpectRejected(short_kv, "KV head array covering one of two layers");

  Metadata uneven = Base();
  uneven[p + "attention.head_count"] = std::uint32_t{5};
  ExpectRejected(uneven, "query heads not divisible by KV heads");
}

void CheckRejections() {
  const std::string p = "kolibri1.";

  Metadata foreign = Base();
  foreign["general.architecture"] = std::string("llama");
  ExpectRejected(foreign, "foreign architecture");

  Metadata half_pattern = Base();
  half_pattern[p + "attention.sliding_window_pattern"] =
      BoolArray{{true, false, true}};
  ExpectRejected(half_pattern, "window pattern for the wrong layer count");

  Metadata no_pattern = Base();
  no_pattern.erase(p + "attention.sliding_window_pattern");
  ExpectRejected(no_pattern, "missing window pattern");

  Metadata global_only = Base();
  global_only[p + "attention.sliding_window_pattern"] =
      BoolArray{{false, false}};
  ExpectRejected(global_only, "model without a sliding layer");

  Metadata odd_head = Base();
  odd_head[p + "attention.value_length"] = std::uint32_t{100};
  odd_head[p + "attention.key_length"] = std::uint32_t{100};
  ExpectRejected(odd_head, "head dimension that is not a multiple of 32");

  Metadata split_heads = Base();
  split_heads[p + "attention.value_length"] = std::uint32_t{96};
  ExpectRejected(split_heads, "key and value head dimensions that differ");

  Metadata zero_window = Base();
  zero_window[p + "attention.sliding_window"] = std::uint32_t{0};
  ExpectRejected(zero_window, "zero sliding window");

  Metadata wide_window = Base();
  wide_window[p + "attention.sliding_window"] = std::uint32_t{8192};
  ExpectRejected(wide_window, "sliding window beyond the context");

  Metadata deepseek = Base();
  deepseek[p + "expert_gating_func"] = std::uint32_t{2};
  ExpectRejected(deepseek,
                 "sigmoid gating that selects on sigmoid(logits)+bias");

  Metadata no_gating = Base();
  no_gating.erase(p + "expert_gating_func");
  ExpectRejected(no_gating, "artifact without a gating rule");

  Metadata renorm = Base();
  renorm[p + "expert_weights_norm"] = true;
  ExpectRejected(renorm, "renormalised expert weights");

  Metadata no_shared = Base();
  no_shared[p + "expert_shared_count"] = std::uint32_t{0};
  ExpectRejected(no_shared, "model without a shared expert");

  Metadata too_many_experts = Base();
  too_many_experts[p + "expert_count"] = std::uint32_t{k1::kMaxExperts + 1};
  ExpectRejected(too_many_experts, "more experts than the kernels bound");

  Metadata too_many_active = Base();
  too_many_active[p + "expert_used_count"] =
      std::uint32_t{k1::kMaxExpertsUsed + 1};
  ExpectRejected(too_many_active, "more active experts than the kernels bound");

  Metadata more_active_than_exist = Base();
  more_active_than_exist[p + "expert_used_count"] = std::uint32_t{9};
  ExpectRejected(more_active_than_exist, "more active experts than exist");
}

std::vector<std::uint8_t> Zeros(std::size_t n) {
  return std::vector<std::uint8_t>(n, 0);
}

/// The shape convention of these artifacts: two dimensions for a plain
/// matrix, a third only where experts are stacked.
std::vector<std::uint64_t> Dims(std::uint64_t cols, std::uint64_t rows,
                                std::uint64_t experts) {
  return experts == 1 ? std::vector<std::uint64_t>{cols, rows}
                      : std::vector<std::uint64_t>{cols, rows, experts};
}

/// F32 payloads: a [cols, rows] matrix, optionally stacked along ne[2], and
/// an [n] vector.
TensorSpec Mat(std::uint64_t cols, std::uint64_t rows,
               std::uint64_t experts = 1) {
  return {Dims(cols, rows, experts), GgmlType::kF32,
          Zeros(cols * rows * experts * 4)};
}
TensorSpec Vec(std::uint64_t n) {
  return {{n}, GgmlType::kF32, Zeros(n * 4)};
}
/// K-quant payloads, optionally stacked along ne[2].
TensorSpec Q4K(std::uint64_t cols, std::uint64_t rows,
               std::uint64_t experts = 1) {
  return {Dims(cols, rows, experts), GgmlType::kQ4_K,
          Zeros(cols / 256 * 144 * rows * experts)};
}
TensorSpec Q6K(std::uint64_t cols, std::uint64_t rows,
               std::uint64_t experts = 1) {
  return {Dims(cols, rows, experts), GgmlType::kQ6_K,
          Zeros(cols / 256 * 210 * rows * experts)};
}
/// A format outside the accepted set; 32-element blocks so the shape checks
/// pass and only the format policy can refuse it.
TensorSpec Q8(std::uint64_t cols, std::uint64_t rows,
              std::uint64_t experts = 1) {
  return {Dims(cols, rows, experts), GgmlType::kQ8_0,
          Zeros(cols / 32 * 34 * rows * experts)};
}

/// The block tensors of every layer. With `quant` the widths of Base() become
/// multiples of 256 through QuantMetadata(), and the K-quant mix of the
/// artifact is reproduced: Q6_K on layer 0's V and down projections.
Tensors Layers(const k1::Config& c, bool quant) {
  const auto matrix = [&](std::uint64_t cols, std::uint64_t rows,
                          std::uint64_t experts = 1) {
    return quant ? Q4K(cols, rows, experts) : Mat(cols, rows, experts);
  };
  Tensors t;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    t[p + "attn_norm.weight"] = Vec(c.hidden_size);
    t[p + "attn_q.weight"] = matrix(c.hidden_size, c.QDim());
    t[p + "attn_k.weight"] = matrix(c.hidden_size, c.KvDim(l));
    t[p + "attn_v.weight"] = quant && l == 0
                                 ? Q6K(c.hidden_size, c.KvDim(l))
                                 : matrix(c.hidden_size, c.KvDim(l));
    t[p + "attn_q_norm.weight"] = Vec(c.head_dim);
    t[p + "attn_k_norm.weight"] = Vec(c.head_dim);
    t[p + "attn_output.weight"] = matrix(c.QDim(), c.hidden_size);
    t[p + "post_attention_norm.weight"] = Vec(c.hidden_size);
    t[p + "ffn_norm.weight"] = Vec(c.hidden_size);
    t[p + "ffn_gate_inp.weight"] = Mat(c.hidden_size, c.num_experts);
    t[p + "exp_probs_b.bias"] = Vec(c.num_experts);
    t[p + "ffn_gate_exps.weight"] =
        matrix(c.hidden_size, c.expert_ffn_size, c.num_experts);
    t[p + "ffn_up_exps.weight"] =
        matrix(c.hidden_size, c.expert_ffn_size, c.num_experts);
    t[p + "ffn_down_exps.weight"] =
        quant && l == 0
            ? Q6K(c.expert_ffn_size, c.hidden_size, c.num_experts)
            : matrix(c.expert_ffn_size, c.hidden_size, c.num_experts);
    t[p + "ffn_gate_shexp.weight"] =
        matrix(c.hidden_size, c.shared_expert_ffn_size);
    t[p + "ffn_up_shexp.weight"] = matrix(c.hidden_size,
                                          c.shared_expert_ffn_size);
    t[p + "ffn_down_shexp.weight"] =
        matrix(c.shared_expert_ffn_size, c.hidden_size);
    t[p + "post_ffw_norm.weight"] = Vec(c.hidden_size);
  }
  return t;
}

Tensors Target(const k1::Config& c, std::uint64_t vocab, bool quant) {
  Tensors t = Layers(c, quant);
  t["token_embd.weight"] =
      quant ? Q4K(c.hidden_size, vocab) : Mat(c.hidden_size, vocab);
  t["output.weight"] =
      quant ? Q6K(c.hidden_size, vocab) : Mat(c.hidden_size, vocab);
  t["output_norm.weight"] = Vec(c.hidden_size);
  return t;
}

void RejectsTensors(const Metadata& meta, const Tensors& tensors,
                    const std::string& what) {
  GgufImage image(meta, tensors);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  std::string error;
  Require(!k1::ModelWeights::Bind(*image.reader(), &error).has_value(),
          what + " accepted");
  Require(!error.empty(), what + " rejected without a message");
}

void CheckWeights() {
  std::string error;
  const auto c = Parse(Base(), &error);
  Require(c.has_value(), error);
  const auto tensors = Target(*c, 96, false);
  GgufImage image(Base(), tensors);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  const auto w = k1::ModelWeights::Bind(*image.reader(), &error);
  Require(w.has_value(), "F32 miniature rejected: " + error);
  Require(w->VocabSize() == 96 && !w->TiedOutput(), "untied LM head");
  Require(w->layers.size() == c->num_layers, "layer count");
  Require(w->layers[0].gate_exps.experts == c->num_experts &&
              w->layers[0].down_exps.experts == c->num_experts,
          "the expert stack must carry every routed expert");
  Require(w->layers[0].gate_exps.ExpertBytes() ==
              static_cast<std::size_t>(c->expert_ffn_size) * c->hidden_size *
                  4,
          "distance between two experts");
  Require(w->layers[0].down_exps.SizeBytes() ==
              static_cast<std::size_t>(c->expert_ffn_size) * c->hidden_size *
                  c->num_experts * 4,
          "expert payload");

  // A tied artifact omits output.weight; the head then is the embedding.
  Tensors tied = tensors;
  tied.erase("output.weight");
  GgufImage tied_image(Base(), tied);
  Require(tied_image.reader() != nullptr, tied_image.error());
  const auto t = k1::ModelWeights::Bind(*tied_image.reader(), &error);
  Require(t.has_value(), "tied artifact rejected: " + error);
  Require(t->TiedOutput(), "the tied head must be the embedding view");

  auto missing = tensors;
  missing.erase("blk.1.attn_v.weight");
  RejectsTensors(Base(), missing, "layer without a V projection");
  missing = tensors;
  missing.erase("blk.0.exp_probs_b.bias");
  RejectsTensors(Base(), missing, "router without its bias");
  missing = tensors;
  missing.erase("blk.0.ffn_down_shexp.weight");
  RejectsTensors(Base(), missing, "layer without the shared expert");

  // Router logits, the expert bias and every norm stay F32 in the artifact;
  // a quantized router would change which experts the top-k picks.
  auto quant_router = tensors;
  quant_router["blk.0.ffn_gate_inp.weight"] = Q8(64, 8);
  RejectsTensors(Base(), quant_router, "quantized router");
  quant_router = tensors;
  quant_router["blk.0.exp_probs_b.bias"] =
      TensorSpec{{64}, GgmlType::kQ8_0, Zeros(64 / 32 * 34)};
  RejectsTensors(Base(), quant_router, "quantized expert bias");

  auto other_format = tensors;
  other_format["blk.0.ffn_up_exps.weight"] = Q8(64, 32, 8);
  RejectsTensors(Base(), other_format, "expert stack in an undecoded format");

  // A payload that runs past the mapped region must never bind, whether the
  // reader catches the short file while parsing or the binder does.
  Tensors short_embd = tensors;
  short_embd["token_embd.weight"] =
      TensorSpec{{64, 96}, GgmlType::kF32, Zeros(64 * 96 * 4 - 128)};
  {
    GgufImage image(Base(), short_embd);
    std::string error;
    Require(image.reader() == nullptr ||
                !k1::ModelWeights::Bind(*image.reader(), &error).has_value(),
            "an embedding past the end of the region bound anyway");
  }

  auto few_experts = tensors;
  few_experts["blk.0.ffn_up_exps.weight"] =
      TensorSpec{{64, 32, 7}, GgmlType::kF32, Zeros(64 * 32 * 7 * 4)};
  RejectsTensors(Base(), few_experts, "expert stack with one expert missing");

  auto wrong_head = tensors;
  wrong_head["output.weight"] = Mat(64, 95);
  RejectsTensors(Base(), wrong_head, "LM head over another vocabulary");
}

/// The released artifact mixes Q4_K and Q6_K inside the same role from layer
/// to layer, and the LM head outranks the embedding. That mix must survive
/// binding instead of being normalised away.
void CheckQuantWeights() {
  std::string error;
  const auto c = Parse(QuantMetadata(), &error);
  Require(c.has_value(), error);
  const auto tensors = Target(*c, 128, true);
  GgufImage image(QuantMetadata(), tensors);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  const auto w = k1::ModelWeights::Bind(*image.reader(), &error);
  Require(w.has_value(), "K-quant miniature rejected: " + error);
  Require(w->layers[0].attn_v.type == GgmlType::kQ6_K &&
              w->layers[1].attn_v.type == GgmlType::kQ4_K,
          "per-layer V formats");
  Require(w->layers[0].down_exps.type == GgmlType::kQ6_K &&
              w->layers[1].down_exps.type == GgmlType::kQ4_K,
          "per-layer down formats");
  Require(w->layers[0].down_exps.ExpertBytes() == 256U * 210,
          "one Q6_K expert of 256 x 256");
  Require(w->layers[1].gate_exps.ExpertBytes() == 256U * 144,
          "one Q4_K expert of 256 x 256");
  Require(w->output.type == GgmlType::kQ6_K &&
              w->token_embd.type == GgmlType::kQ4_K,
          "the LM head keeps a higher quant than the embedding");

  auto q4_0 = tensors;
  q4_0["blk.0.ffn_up_exps.weight"] = TensorSpec{
      {256, 256, 8}, GgmlType::kQ4_0, Zeros(256 / 32 * 16 * 256 * 8)};
  RejectsTensors(QuantMetadata(), q4_0, "Q4_0 expert stack");
}

}  // namespace

int main() {
  return kolibri1_test::Run([] {
    CheckTarget();
    CheckArtifactGeometry();
    CheckPerLayerValues();
    CheckRejections();
    CheckWeights();
    CheckQuantWeights();
  });
}
