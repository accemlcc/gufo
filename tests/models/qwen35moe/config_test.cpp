#include "src/models/qwen35moe/config.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include "tests/models/qwen35moe/check.hpp"
#include "tests/models/qwen35moe/gguf_builder.hpp"

namespace q35 = gufo::models::qwen35moe;
using qwen35moe_test::BoolArray;
using qwen35moe_test::GgufImage;
using qwen35moe_test::Metadata;
using qwen35moe_test::Require;
using qwen35moe_test::U64Array;

/// A four-layer trunk miniature (three linear, one full) plus the MTP block,
/// with the routing and SSM metadata of the released files.
Metadata Base() {
  const std::string p("qwen35moe.");
  return {
      {"general.architecture", std::string("qwen35moe")},
      {p + "block_count", std::uint32_t{5}},
      {p + "nextn_predict_layers", std::uint32_t{1}},
      {p + "embedding_length", std::uint32_t{64}},
      {p + "context_length", std::uint32_t{4096}},
      {p + "attention.layer_norm_rms_epsilon", 1e-6F},
      {p + "attention.head_count", std::uint32_t{4}},
      {p + "attention.head_count_kv", std::uint32_t{2}},
      {p + "attention.key_length", std::uint32_t{32}},
      {p + "attention.value_length", std::uint32_t{32}},
      {p + "rope.dimension_count", std::uint32_t{16}},
      {p + "rope.dimension_sections", U64Array{{3, 3, 2, 0}}},
      {p + "rope.freq_base", 10000000.0F},
      {p + "full_attention_interval", std::uint32_t{4}},
      {p + "attention.recurrent_layers",
       BoolArray{{true, true, true, false, false}}},
      {p + "ssm.conv_kernel", std::uint32_t{4}},
      {p + "ssm.state_size", std::uint32_t{16}},
      {p + "ssm.group_count", std::uint32_t{2}},
      {p + "ssm.time_step_rank", std::uint32_t{2}},
      {p + "ssm.inner_size", std::uint32_t{32}},
      {p + "expert_count", std::uint32_t{8}},
      {p + "expert_used_count", std::uint32_t{2}},
      {p + "expert_feed_forward_length", std::uint32_t{32}},
      {p + "expert_shared_feed_forward_length", std::uint32_t{32}},
  };
}

std::optional<q35::Config> Parse(const Metadata& metadata, std::string* error) {
  GgufImage image(metadata);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  return q35::Config::FromGguf(*image.reader(), error);
}

void ValidMiniature() {
  std::string error;
  const auto config = Parse(Base(), &error);
  Require(config.has_value(), "valid miniature rejected: " + error);
  Require(config->num_layers == 4, "trunk is block_count minus nextn");
  Require(config->hidden_size == 64, "hidden size");
  Require(config->context_length == 4096, "context");
  Require(config->has_mtp, "MTP block detected");
  Require(config->LinearLayerCount() == 3, "three linear layers");
  Require(config->FullLayerCount() == 1, "one full layer");
  Require(!config->IsLinear(3), "layer 3 is the full layer");
  Require(config->QProjDim() == 2 * 4 * 32, "packed q|gate width");
  Require(config->KvDim() == 2 * 32, "KV width");
  Require(config->SsmQkvDim() == 2 * (2 * 16) + 32, "SSM qkv width");
  Require(config->SsmValueHeadDim() == 16, "SSM value head dim");
  Require(config->num_experts == 8 && config->experts_used == 2, "routing");
  Require(config->gating == q35::ExpertGating::kSoftmaxTopKRenorm,
          "softmax top-k rule");
}

void WrongArchitecture() {
  Metadata m = Base();
  m["general.architecture"] = std::string("qwen3moe");
  std::string error;
  Require(!Parse(m, &error).has_value(), "foreign architecture accepted");
  Require(!error.empty(), "rejection names no reason");
}

void PatternMustMatchInterval() {
  Metadata m = Base();
  m["qwen35moe.attention.recurrent_layers"] =
      BoolArray{{true, true, true, true, false}};
  std::string error;
  Require(!Parse(m, &error).has_value(), "off-pattern layer accepted");
}

void PatternFallsBackToInterval() {
  // Older converters omit the key; the interval rule decides the kinds.
  Metadata m = Base();
  m.erase("qwen35moe.attention.recurrent_layers");
  std::string error;
  const auto config = Parse(m, &error);
  Require(config.has_value(), "interval fallback rejected: " + error);
  Require(config->LinearLayerCount() == 3 && config->FullLayerCount() == 1,
          "fallback pattern is 3:1");
}

void MtpBlockMustBeFull() {
  Metadata m = Base();
  m["qwen35moe.attention.recurrent_layers"] =
      BoolArray{{true, true, true, false, true}};
  std::string error;
  Require(!Parse(m, &error).has_value(), "recurrent MTP block accepted");
}

void SsmMustFactor() {
  Metadata m = Base();
  m["qwen35moe.ssm.inner_size"] = std::uint32_t{33};
  std::string error;
  Require(!Parse(m, &error).has_value(), "unfactorable SSM accepted");
}

void ExpertsBounded() {
  {
    Metadata m = Base();
    m["qwen35moe.expert_used_count"] = std::uint32_t{9};
    std::string error;
    Require(!Parse(m, &error).has_value(), "used > experts accepted");
  }
  {
    Metadata m = Base();
    m["qwen35moe.expert_count"] = std::uint32_t{256};
    m["qwen35moe.expert_used_count"] = std::uint32_t{17};
    std::string error;
    Require(!Parse(m, &error).has_value(), "used > kMaxExpertsUsed accepted");
  }
}

void RopeSectionsMustSum() {
  Metadata m = Base();
  m["qwen35moe.rope.dimension_sections"] = U64Array{{3, 3, 3, 0}};
  std::string error;
  Require(!Parse(m, &error).has_value(), "bad mRoPE sections accepted");
}

int main() {
  return qwen35moe_test::Run([] {
    ValidMiniature();
    WrongArchitecture();
    PatternMustMatchInterval();
    PatternFallsBackToInterval();
    MtpBlockMustBeFull();
    SsmMustFactor();
    ExpertsBounded();
    RopeSectionsMustSum();
  });
}
