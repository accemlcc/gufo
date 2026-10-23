// Binds the real qwen35moe artifact named by GUFO_QWEN35MOE_MODEL and pins
// the geometry of the released Ornith-1.5 files, so a new quant or a changed
// converter shows up as a failing check instead of a surprise at the first
// decode. The multi-shard 122B file loads through the same split reader the
// server will use.
//
// Set GUFO_QWEN35MOE_MODEL to a qwen35moe GGUF (first shard for splits);
// without it the check skips (77) and the hosted suite never needs the file.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "src/models/qwen35moe/weights.hpp"
#include "tests/models/qwen35moe/check.hpp"

namespace q35 = gufo::models::qwen35moe;
using gufo::core::GgmlType;
using qwen35moe_test::Require;

namespace {

void CheckVariant(const q35::Config& c) {
  Require(c.context_length == 262144, "262144 native context");
  Require(c.num_kv_heads == 2 && c.head_dim == 256, "GQA head geometry");
  Require(c.rotary_dim == 64, "64-wide partial RoPE");
  Require(c.rope_sections[0] == 11 && c.rope_sections[1] == 11 &&
              c.rope_sections[2] == 10 && c.rope_sections[3] == 0,
          "mRoPE sections");
  Require(c.full_attention_interval == 4, "3:1 linear-to-full ratio");
  Require(c.LinearLayerCount() * 4 == 3 * c.num_layers, "3:1 layer mix");
  Require(c.num_experts == 256 && c.experts_used == 8, "8 of 256 experts");
  Require(c.has_mtp, "the released files carry the nextn block");
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    Require(c.IsLinear(l) == (((l + 1) % 4) != 0), "interval pattern");
  }
  if (c.hidden_size == 2048) {
    Require(c.num_layers == 40, "35B trunk depth");
    Require(c.num_heads == 16, "35B heads");
    Require(c.ssm_group_count == 16 && c.ssm_time_step_rank == 32 &&
                c.ssm_inner_size == 4096 && c.ssm_state_size == 128 &&
                c.ssm_conv_kernel == 4,
            "35B SSM dims");
    Require(c.expert_ffn_size == 512 && c.shared_ffn_size == 512,
            "35B expert width");
  } else if (c.hidden_size == 3072) {
    Require(c.num_layers == 48, "122B trunk depth");
    Require(c.num_heads == 32, "122B heads");
    Require(c.ssm_group_count == 16 && c.ssm_time_step_rank == 64 &&
                c.ssm_inner_size == 8192 && c.ssm_state_size == 128 &&
                c.ssm_conv_kernel == 4,
            "122B SSM dims");
    Require(c.expert_ffn_size == 1024 && c.shared_ffn_size == 1024,
            "122B expert width");
  } else {
    Require(false, "unknown qwen35moe width");
  }
}

void CheckLayout(const q35::ModelWeights& w) {
  const auto expect_float = [&](const q35::TensorRef& t,
                                const std::string& name) {
    Require(t.type == GgmlType::kF32, name + " stays in F32");
  };
  for (const auto& layer : w.layers) {
    expect_float(layer.attn_norm, "attn_norm");
    expect_float(layer.post_attention_norm, "post_attention_norm");
    expect_float(layer.ffn_gate_inp, "router");
    expect_float(layer.ffn_gate_inp_shexp, "shared-expert gate");
    if (layer.is_full_attention) {
      expect_float(layer.attn_q_norm, "q norm");
      expect_float(layer.attn_k_norm, "k norm");
    } else {
      expect_float(layer.ssm_a, "ssm_a");
      expect_float(layer.ssm_dt, "ssm_dt");
    }
    Require(!layer.ffn_gate_exps.empty() && !layer.ffn_up_exps.empty() &&
                !layer.ffn_down_exps.empty(),
            "routed experts bind in every trunk layer");
  }
  expect_float(w.output_norm, "output_norm");
  Require(!w.token_embd.empty() && !w.output.empty(), "embedding and head");
}

}  // namespace

int main() {
  return qwen35moe_test::Run([] {
    const char* path = std::getenv("GUFO_QWEN35MOE_MODEL");
    if (path == nullptr || path[0] == '\0') {
      std::cout << "SKIP: GUFO_QWEN35MOE_MODEL is not set\n";
      std::exit(77);
    }
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "cannot open artifact: " + error);
    const auto weights = q35::ModelWeights::LoadFromGguf(*reader, &error);
    Require(weights.has_value(), "artifact rejected: " + error);
    CheckVariant(weights->config);
    CheckLayout(*weights);
    std::cout << "bound trunk=" << weights->config.num_layers
              << " hidden=" << weights->config.hidden_size
              << " mtp=" << (weights->config.has_mtp ? "yes" : "no") << '\n';
  });
}
