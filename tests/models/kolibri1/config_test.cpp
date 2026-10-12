#include "src/models/kolibri1/config.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// The in-memory GGUF writer and the always-on assertion are shared with the
// gemma4 host checks rather than duplicated per model.
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/gguf_builder.hpp"

namespace k1 = gufo::models::kolibri1;
using gemma4_test::BoolArray;
using gemma4_test::GgufImage;
using gemma4_test::I32Array;
using gemma4_test::Metadata;
using gemma4_test::Require;

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

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckTarget();
    CheckArtifactGeometry();
    CheckPerLayerValues();
    CheckRejections();
  });
}
