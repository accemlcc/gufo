#include "src/models/qwen35moe/weights.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/models/qwen35moe/check.hpp"
#include "tests/models/qwen35moe/gguf_builder.hpp"

namespace q35 = gufo::models::qwen35moe;
using gufo::core::GgmlType;
using qwen35moe_test::BoolArray;
using qwen35moe_test::GgufImage;
using qwen35moe_test::Metadata;
using qwen35moe_test::Require;
using qwen35moe_test::Tensors;
using qwen35moe_test::TensorSpec;
using qwen35moe_test::U64Array;

constexpr std::uint32_t kHidden = 64;
constexpr std::uint32_t kVocab = 128;

Metadata Base() {
  const std::string p("qwen35moe.");
  return {
      {"general.architecture", std::string("qwen35moe")},
      {p + "block_count", std::uint32_t{5}},
      {p + "nextn_predict_layers", std::uint32_t{1}},
      {p + "embedding_length", std::uint32_t{kHidden}},
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

void AddF32(Tensors& tensors, const std::string& name, std::size_t elements) {
  tensors[name] = TensorSpec{{elements}, GgmlType::kF32,
                             std::vector<std::uint8_t>(4 * elements, 0)};
}

void AddLayer(Tensors& tensors, const std::string& prefix, bool full) {
  AddF32(tensors, prefix + "attn_norm.weight", kHidden);
  AddF32(tensors, prefix + "post_attention_norm.weight", kHidden);
  if (full) {
    AddF32(tensors, prefix + "attn_q.weight", kHidden * 256);
    AddF32(tensors, prefix + "attn_k.weight", kHidden * 64);
    AddF32(tensors, prefix + "attn_v.weight", kHidden * 64);
    AddF32(tensors, prefix + "attn_output.weight", 128 * kHidden);
    AddF32(tensors, prefix + "attn_q_norm.weight", 32);
    AddF32(tensors, prefix + "attn_k_norm.weight", 32);
  } else {
    AddF32(tensors, prefix + "attn_qkv.weight", kHidden * 96);
    AddF32(tensors, prefix + "attn_gate.weight", kHidden * 32);
    AddF32(tensors, prefix + "ssm_a", 2);
    AddF32(tensors, prefix + "ssm_conv1d.weight", 4 * 96);
    AddF32(tensors, prefix + "ssm_dt.bias", 2);
    AddF32(tensors, prefix + "ssm_alpha.weight", kHidden * 2);
    AddF32(tensors, prefix + "ssm_beta.weight", kHidden * 2);
    AddF32(tensors, prefix + "ssm_norm.weight", 16);
    AddF32(tensors, prefix + "ssm_out.weight", 32 * kHidden);
  }
  AddF32(tensors, prefix + "ffn_gate_inp.weight", kHidden * 8);
  AddF32(tensors, prefix + "ffn_gate_inp_shexp.weight", kHidden);
  AddF32(tensors, prefix + "ffn_gate_shexp.weight", kHidden * 32);
  AddF32(tensors, prefix + "ffn_up_shexp.weight", kHidden * 32);
  AddF32(tensors, prefix + "ffn_down_shexp.weight", 32 * kHidden);
  AddF32(tensors, prefix + "ffn_gate_exps.weight", kHidden * 32 * 8);
  AddF32(tensors, prefix + "ffn_up_exps.weight", kHidden * 32 * 8);
  AddF32(tensors, prefix + "ffn_down_exps.weight", 32 * kHidden * 8);
}

Tensors FullTensors() {
  Tensors tensors;
  AddF32(tensors, "token_embd.weight", kHidden * kVocab);
  AddF32(tensors, "output_norm.weight", kHidden);
  AddF32(tensors, "output.weight", kHidden * kVocab);
  for (int i = 0; i < 4; ++i) {
    AddLayer(tensors, "blk." + std::to_string(i) + ".", i == 3);
  }
  AddLayer(tensors, "blk.4.", true);
  AddF32(tensors, "blk.4.nextn.eh_proj.weight", 2 * kHidden * kHidden);
  AddF32(tensors, "blk.4.nextn.enorm.weight", kHidden);
  AddF32(tensors, "blk.4.nextn.hnorm.weight", kHidden);
  AddF32(tensors, "blk.4.nextn.shared_head_norm.weight", kHidden);
  return tensors;
}

std::optional<q35::ModelWeights> Bind(const Metadata& metadata,
                                      const Tensors& tensors,
                                      std::string* error) {
  GgufImage image(metadata, tensors);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  return q35::ModelWeights::LoadFromGguf(*image.reader(), error);
}

void ValidBind() {
  std::string error;
  const auto weights = Bind(Base(), FullTensors(), &error);
  Require(weights.has_value(), "valid miniature rejected: " + error);
  Require(weights->config.num_layers == 4, "trunk depth");
  Require(weights->layers.size() == 4, "four trunk layers");
  Require(!weights->layers[0].is_full_attention, "layer 0 linear");
  Require(weights->layers[3].is_full_attention, "layer 3 full");
  Require(!weights->layers[3].attn_q.empty(), "packed q|gate binds");
  Require(!weights->layers[0].attn_qkv.empty(), "SSM qkv binds");
  Require(!weights->layers[2].ffn_down_exps.empty(), "routed experts bind");
  Require(!weights->mtp.block.attn_q.empty(), "MTP block binds");
  Require(!weights->mtp.eh_proj.empty(), "MTP projection binds");
}

void TiedOutput() {
  Tensors tensors = FullTensors();
  tensors.erase("output.weight");
  std::string error;
  const auto weights = Bind(Base(), tensors, &error);
  Require(weights.has_value(), "tied output rejected: " + error);
  Require(weights->output.data == weights->token_embd.data,
          "tied output shares the embedding");
}

void MissingTensor() {
  Tensors tensors = FullTensors();
  tensors.erase("blk.2.ssm_out.weight");
  std::string error;
  Require(!Bind(Base(), tensors, &error).has_value(), "missing tensor bound");
  Require(error.find("blk.2.ssm_out.weight") != std::string::npos,
          "rejection names the tensor: " + error);
}

void NormMustBeFloat() {
  Tensors tensors = FullTensors();
  tensors["blk.1.attn_norm.weight"] =
      TensorSpec{{kHidden}, GgmlType::kQ8_0,
                 std::vector<std::uint8_t>(34 * (kHidden / 32), 0)};
  std::string error;
  Require(!Bind(Base(), tensors, &error).has_value(), "packed norm bound");
  Require(error.find("blk.1.attn_norm.weight") != std::string::npos,
          "rejection names the tensor: " + error);
}

void ShapeMismatch() {
  Tensors tensors = FullTensors();
  tensors["blk.3.attn_q.weight"] =
      TensorSpec{{kHidden * 128}, GgmlType::kF32,
                 std::vector<std::uint8_t>(4 * kHidden * 128, 0)};
  std::string error;
  Require(!Bind(Base(), tensors, &error).has_value(), "short tensor bound");
}

int main() {
  return qwen35moe_test::Run([] {
    ValidBind();
    TiedOutput();
    MissingTensor();
    NormMustBeFloat();
    ShapeMismatch();
  });
}
