#include "src/models/qwen35moe/config.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::qwen35moe {
namespace {

constexpr std::string_view kArchitecture = "qwen35moe";

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

  /// Integer arrays of any width; GGUF writers pick the narrowest type, and
  /// boolean arrays decode to 0/1 the same way.
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
};

}  // namespace

std::uint32_t Config::LinearLayerCount() const noexcept {
  std::uint32_t count = 0;
  for (auto r : recurrent) {
    count += r != 0 ? 1U : 0U;
  }
  return count;
}

std::optional<Config> Config::FromGguf(const core::GgufReader& reader,
                                       std::string* error_msg) {
  Reader r{reader, std::string(kArchitecture) + ".", error_msg};
  if (reader.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not qwen35moe");
    return std::nullopt;
  }

  Config c;
  const std::uint32_t block_count = r.U32("block_count");
  const std::uint32_t nextn = r.U32("nextn_predict_layers", false, 0);
  if (!r.ok) {
    return std::nullopt;
  }
  if (block_count < 2 || nextn > 1 || nextn >= block_count) {
    r.Fail("qwen35moe needs a trunk of at least one layer");
    return std::nullopt;
  }
  c.num_layers = block_count - nextn;
  c.has_mtp = nextn == 1;
  c.hidden_size = r.U32("embedding_length");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");
  c.num_heads = r.U32("attention.head_count");
  c.num_kv_heads = r.U32("attention.head_count_kv");
  c.head_dim = r.U32("attention.key_length");
  if (r.ok && r.U32("attention.value_length") != c.head_dim) {
    r.Fail("qwen35moe requires equal key and value head dimensions");
  }
  c.rotary_dim = r.U32("rope.dimension_count");
  c.rope_theta = r.F32("rope.freq_base");
  c.full_attention_interval = r.U32("full_attention_interval");
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.hidden_size == 0 || c.context_length == 0 || !(c.rms_eps > 0.0F) ||
      c.num_heads == 0 || c.num_kv_heads == 0 || c.head_dim == 0 ||
      c.rotary_dim == 0 || !(c.rope_theta > 0.0F) ||
      c.full_attention_interval == 0) {
    r.Fail("qwen35moe dimensions must be positive");
    return std::nullopt;
  }
  if (c.num_heads % c.num_kv_heads != 0) {
    r.Fail("qwen35moe query heads must be a multiple of KV heads");
    return std::nullopt;
  }
  if (c.rotary_dim % 2 != 0 || c.rotary_dim > c.head_dim) {
    r.Fail("qwen35moe rotary dimension must be even and within the head");
    return std::nullopt;
  }
  // mRoPE: three temporal/spatial axes plus the zero pad. The section widths
  // sum to half the rotary dimension.
  const auto sections = r.U64Array("rope.dimension_sections");
  if (r.ok) {
    if (sections.size() != 4 || sections[3] != 0) {
      r.Fail("rope.dimension_sections must contain three axes and a zero pad");
    } else {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < 3; ++i) {
        if (sections[i] == 0 || sections[i] > c.rotary_dim / 2) {
          r.Fail("invalid mRoPE axis dimension");
          break;
        }
        sum += sections[i];
        c.rope_sections[i] = static_cast<std::uint32_t>(sections[i]);
      }
      c.rope_sections[3] = 0;
      if (r.ok && sum != c.rotary_dim / 2) {
        r.Fail("mRoPE sections must sum to half the rotary dimension");
      }
    }
  }
  if (!r.ok) {
    return std::nullopt;
  }

  // Layer pattern: one boolean per block, trunk first, the MTP block last.
  // Older converters omit the key; the interval rule below is then the
  // pattern, and the weight binder cross-checks it against the tensors of
  // every layer, so a file that disagrees with itself still fails.
  const bool has_pattern =
      reader.FindMetadata(r.Key("attention.recurrent_layers")) != nullptr;
  std::vector<std::uint64_t> pattern;
  if (has_pattern) {
    pattern = r.U64Array("attention.recurrent_layers");
    if (r.ok && pattern.size() != block_count) {
      r.Fail("attention.recurrent_layers must describe every block");
    }
    if (!r.ok) {
      return std::nullopt;
    }
    for (auto v : pattern) {
      if (v > 1) {
        r.Fail("attention.recurrent_layers must be boolean");
        return std::nullopt;
      }
    }
    if (c.has_mtp && pattern.back() != 0) {
      r.Fail("the nextn predictor block must be a full-attention layer");
      return std::nullopt;
    }
  }
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const bool expect_linear = ((l + 1) % c.full_attention_interval) != 0;
    if (has_pattern && (pattern[l] != 0) != expect_linear) {
      r.Fail("layer " + std::to_string(l) +
             " disagrees with full_attention_interval");
      return std::nullopt;
    }
    c.recurrent.push_back(expect_linear ? 1 : 0);
  }
  if (!r.ok) {
    return std::nullopt;
  }

  c.ssm_conv_kernel = r.U32("ssm.conv_kernel");
  c.ssm_state_size = r.U32("ssm.state_size");
  c.ssm_group_count = r.U32("ssm.group_count");
  c.ssm_time_step_rank = r.U32("ssm.time_step_rank");
  c.ssm_inner_size = r.U32("ssm.inner_size");
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.ssm_conv_kernel == 0 || c.ssm_state_size == 0 ||
      c.ssm_group_count == 0 || c.ssm_time_step_rank == 0 ||
      c.ssm_inner_size == 0) {
    r.Fail("qwen35moe SSM dimensions must be positive");
    return std::nullopt;
  }
  if (c.ssm_inner_size != c.ssm_time_step_rank * c.ssm_state_size) {
    r.Fail("ssm.inner_size must factor into rank times state size");
    return std::nullopt;
  }

  c.num_experts = r.U32("expert_count");
  c.experts_used = r.U32("expert_used_count");
  c.expert_ffn_size = r.U32("expert_feed_forward_length");
  c.shared_ffn_size = r.U32("expert_shared_feed_forward_length");
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.num_experts == 0 || c.num_experts > kMaxExperts) {
    r.Fail("qwen35moe needs 1.." + std::to_string(kMaxExperts) +
           " routed experts per layer");
    return std::nullopt;
  }
  if (c.experts_used == 0 || c.experts_used > c.num_experts ||
      c.experts_used > kMaxExpertsUsed) {
    r.Fail("qwen35moe routing must use 1.." +
           std::to_string(kMaxExpertsUsed) + " experts of " +
           std::to_string(c.num_experts));
    return std::nullopt;
  }
  if (c.expert_ffn_size == 0 || c.shared_ffn_size == 0) {
    r.Fail("qwen35moe needs routed experts and a shared expert per layer");
    return std::nullopt;
  }
  c.gating = ExpertGating::kSoftmaxTopKRenorm;
  return c;
}

}  // namespace gufo::models::qwen35moe
