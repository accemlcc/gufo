#include "src/models/kolibri1/config.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::kolibri1 {
namespace {

constexpr std::string_view kArchitecture = "kolibri1";

/// `expert_gating_func` value llama.cpp writes for the sigmoid rule that
/// selects on `logits + exp_probs_b` and weights by `sigmoid(logits)`.
constexpr std::uint32_t kGatingSigmoidLogitAdd = 5;

struct Reader {
  const core::GgufReader& gguf;
  std::string prefix;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  [[nodiscard]] std::string Key(std::string_view suffix) const {
    return prefix + std::string(suffix);
  }

  std::uint32_t U32(std::string_view suffix, bool required = true,
                    std::uint32_t fallback = 0) {
    const std::string key = Key(suffix);
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + key);
    }
    return fallback;
  }

  float F32(std::string_view suffix, bool required = true,
            float fallback = 0.0F) {
    const std::string key = Key(suffix);
    const auto value = gguf.GetMetadataFloat32(key);
    if (value.has_value() && std::isfinite(*value)) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF float " + key);
    }
    return fallback;
  }

  bool Bool(std::string_view suffix, bool required = true,
            bool fallback = false) {
    const std::string key = Key(suffix);
    const auto value = gguf.GetMetadataBool(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF boolean " + key);
    }
    return fallback;
  }

  /// Integer or boolean arrays of any width; GGUF writers pick the type.
  std::vector<std::uint64_t> U64Array(std::string_view suffix) {
    const std::string key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + key);
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + key);
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + key);
    }
    return out;
  }

  /// Per-layer values stored either as one scalar or as one entry per layer.
  std::vector<std::uint32_t> PerLayer(std::string_view suffix,
                                      std::uint32_t layers) {
    const std::string key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    if (meta == nullptr) {
      Fail("missing GGUF value " + key);
      return {};
    }
    if (const auto scalar = gguf.GetMetadataUint32(key)) {
      return std::vector<std::uint32_t>(layers, *scalar);
    }
    const auto values = U64Array(suffix);
    if (!ok) {
      return {};
    }
    if (values.size() != layers) {
      Fail(key + " must describe every layer");
      return {};
    }
    std::vector<std::uint32_t> out;
    for (auto v : values) {
      if (v == 0 || v > 0xffffffffULL) {
        Fail(key + " has an invalid entry");
        return {};
      }
      out.push_back(static_cast<std::uint32_t>(v));
    }
    return out;
  }
};

}  // namespace

std::uint32_t Config::GlobalLayerCount() const noexcept {
  std::uint32_t count = 0;
  for (auto s : sliding) {
    count += s == 0 ? 1U : 0U;
  }
  return count;
}

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, std::string(kArchitecture) + ".", error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not kolibri1");
    return std::nullopt;
  }

  Config c;
  c.num_layers = r.U32("block_count");
  c.hidden_size = r.U32("embedding_length");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");
  c.num_heads = r.U32("attention.head_count");
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.num_layers == 0 || c.hidden_size == 0 || c.num_heads == 0 ||
      c.context_length == 0 || !(c.rms_eps > 0.0F)) {
    r.Fail("kolibri1 dimensions must be positive");
    return std::nullopt;
  }

  c.kv_heads = r.PerLayer("attention.head_count_kv", c.num_layers);

  // The artifact stores the pattern as one boolean per layer; Kolibri uses a
  // 4:1 sliding-to-global ratio, but any mix that keeps both kinds works.
  const auto pattern = r.U64Array("attention.sliding_window_pattern");
  if (r.ok && pattern.size() != c.num_layers) {
    r.Fail("attention.sliding_window_pattern must describe every layer");
  }
  // A pattern that marks no layer sliding is not something this engine runs:
  // the sliding rings, and the boundary snapshots snap down to, are built from
  // the sliding layers, and no artifact without one has ever been computed.
  // Checking only the length lets an all-global pattern through, which is what
  // it took upstream's 8- and 16-bit array decoding (#532) to expose: before
  // that, the boolean pattern read as an empty array and the length check
  // rejected it for the wrong reason.
  if (r.ok && !std::any_of(pattern.begin(), pattern.end(),
                           [](std::uint64_t value) { return value != 0; })) {
    r.Fail("attention.sliding_window_pattern marks no sliding layer");
  }
  if (!r.ok) {
    return std::nullopt;
  }
  for (auto v : pattern) {
    c.sliding.push_back(v != 0 ? 1 : 0);
  }

  // One head dimension for both layer kinds: Kolibri's global layers differ
  // from its sliding layers only in the window and in carrying no positional
  // encoding. A writer that splits the key and value widths per kind is only
  // accepted when both agree.
  c.head_dim = r.U32("attention.key_length");
  if (r.ok && r.U32("attention.value_length") != c.head_dim) {
    r.Fail("kolibri1 requires equal query and key head dimensions");
  }
  for (std::string_view suffix :
       {"attention.key_length_swa", "attention.value_length_swa"}) {
    if (r.ok && gguf.FindMetadata(r.Key(suffix)) != nullptr &&
        r.U32(suffix) != c.head_dim) {
      r.Fail("kolibri1 uses one head dimension for both layer kinds");
    }
  }
  c.rope_theta = r.F32("rope.freq_base");
  c.sliding_window = r.U32("attention.sliding_window");
  if (!r.ok) {
    return std::nullopt;
  }

  // Kernels reduce a head in 32-lane groups, and the sliding window has to
  // address keys inside the trained range.
  if (c.head_dim == 0 || c.head_dim % 32 != 0) {
    r.Fail("kolibri1 head dimension must be a positive multiple of 32");
    return std::nullopt;
  }
  if (!(c.rope_theta > 0.0F)) {
    r.Fail("kolibri1 rope base must be positive");
    return std::nullopt;
  }
  if (c.sliding_window == 0 || c.sliding_window > c.context_length) {
    r.Fail("kolibri1 sliding window must be positive and within the context");
    return std::nullopt;
  }
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (c.num_heads % c.kv_heads[l] != 0) {
      r.Fail("kolibri1 query heads must be a multiple of KV heads in layer " +
             std::to_string(l));
      return std::nullopt;
    }
  }

  // Routing. Every layer routes `expert_used_count` of `expert_count` and
  // adds one ungated shared expert; the weights are the unbiased sigmoid of
  // the router logits.
  c.num_experts = r.U32("expert_count");
  c.experts_used = r.U32("expert_used_count");
  c.expert_ffn_size = r.U32("expert_feed_forward_length");
  c.shared_expert_count = r.U32("expert_shared_count");
  c.shared_expert_ffn_size = r.U32("expert_shared_feed_forward_length");
  c.renormalize_expert_weights = r.Bool("expert_weights_norm");
  const auto gating = r.U32("expert_gating_func");
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.num_experts == 0 || c.num_experts > kMaxExperts) {
    r.Fail("kolibri1 needs 1.." + std::to_string(kMaxExperts) +
           " routed experts per layer");
    return std::nullopt;
  }
  if (c.experts_used == 0 || c.experts_used > c.num_experts ||
      c.experts_used > kMaxExpertsUsed) {
    r.Fail("kolibri1 routing must use 1.." + std::to_string(kMaxExpertsUsed) +
           " experts of " + std::to_string(c.num_experts));
    return std::nullopt;
  }
  if (c.expert_ffn_size == 0 || c.shared_expert_count == 0 ||
      c.shared_expert_ffn_size == 0) {
    r.Fail("kolibri1 needs routed experts and one shared expert per layer");
    return std::nullopt;
  }
  if (gating != kGatingSigmoidLogitAdd) {
    r.Fail("kolibri1 routing must use sigmoid_logit_add gating "
           "(expert_gating_func 5)");
    return std::nullopt;
  }
  c.gating = ExpertGating::kSigmoidLogitAdd;
  if (c.renormalize_expert_weights) {
    r.Fail("kolibri1 renormalised expert weights are not implemented");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::kolibri1
