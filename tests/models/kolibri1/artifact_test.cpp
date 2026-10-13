// Binds the real Kolibri 1 artifact named by GUFO_KOLIBRI1_MODEL and pins what
// the released quantization looks like, so a new quant or a changed converter
// shows up as a failing census instead of a surprise at the first decode.
//
// Set GUFO_KOLIBRI1_MODEL to a Kolibri-1 GGUF; without it the check skips (77)
// and the hosted suite never needs the file.
#include "src/models/kolibri1/chat_template.hpp"
#include "src/models/kolibri1/weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "tests/models/kolibri1/check.hpp"

namespace k1 = gufo::models::kolibri1;
using kolibri1_test::Require;
using gufo::core::GgmlType;

namespace {

/// Number of layers whose value projection (and the two down projections that
/// share its fate) keep the higher quant in the released Q4_K_M file.
constexpr std::uint32_t kHighQuantLayers = 25;

[[nodiscard]] std::uint32_t Count(
    const k1::ModelWeights& w, GgmlType type,
    const k1::TensorRef k1::LayerWeights::*role) {
  std::uint32_t count = 0;
  for (const auto& layer : w.layers) {
    count += (layer.*role).type == type ? 1U : 0U;
  }
  return count;
}

void CheckConfig(const k1::Config& c) {
  Require(c.num_layers == 50 && c.hidden_size == 2560 &&
              c.context_length == 262144,
          "the released artifact has 50 layers, is 2560 wide and trained to "
          "262144 positions");
  Require(c.num_heads == 48 && c.head_dim == 128, "48 query heads of 128");
  Require(c.sliding_window == 513, "the sliding window holds 513 keys");
  Require(c.SlidingLayerCount() == 40 && c.GlobalLayerCount() == 10,
          "the 4:1 pattern");
  Require(c.num_experts == 384 && c.experts_used == 6, "6 of 384 experts");
  Require(c.expert_ffn_size == 512 && c.shared_expert_count == 1 &&
              c.shared_expert_ffn_size == 512,
          "one shared expert of the routed width");
  Require(c.gating == k1::ExpertGating::kSigmoidLogitAdd &&
              !c.renormalize_expert_weights,
          "sigmoid_logit_add routing without renormalisation");

  // The converter writes the pattern as four sliding layers per global one,
  // and the global layers carry no positional encoding at all.
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    Require(c.IsSliding(l) == ((l + 1) % 5 != 0),
            "the sliding pattern must repeat every five layers");
    Require(c.UsesRope(l) == c.IsSliding(l) &&
                c.RopeDim(l) == (c.IsSliding(l) ? 128U : 0U),
            "rope runs in the sliding layers only");
  }
}

void CheckLayout(const k1::ModelWeights& w) {
  Require(w.VocabSize() == 128000, "vocabulary");
  Require(!w.TiedOutput(), "the released head is untied");
  Require(w.token_embd.type == GgmlType::kQ4_K &&
              w.output.type == GgmlType::kQ6_K,
          "the LM head outranks the embedding in the Q4_K_M recipe");
  Require(w.output_norm.type == GgmlType::kF32, "output norm");
  for (std::uint32_t l = 0; l < w.config.num_layers; ++l) {
    const auto& layer = w.layers[l];
    Require(layer.attn_q.type == GgmlType::kQ4_K &&
                layer.attn_k.type == GgmlType::kQ4_K &&
                layer.attn_output.type == GgmlType::kQ4_K,
            "attention projections are Q4_K in every layer");
    Require(layer.gate_exps.type == GgmlType::kQ4_K &&
                layer.up_exps.type == GgmlType::kQ4_K,
            "the gate and up expert stacks are Q4_K in every layer");
    Require(layer.router.type == GgmlType::kF32 &&
                layer.router_bias.type == GgmlType::kF32,
            "routing stays in F32");
    Require(layer.attn_q_norm.cols == 128 && layer.attn_k_norm.cols == 128,
            "one q/k norm vector per layer, shared by its heads");
    Require(layer.gate_exps.experts == 384 && layer.up_exps.experts == 384 &&
                layer.down_exps.experts == 384,
            "every expert stack holds all 384 experts");
    Require(layer.router.cols == 2560 && layer.router.rows == 384 &&
                layer.router_bias.cols == 384,
            "the router reads the hidden width and scores every expert");
  }
}

/// The three roles the converter upgrades together: a layer either keeps Q6_K
/// in all of them or runs all of them in Q4_K.
void CheckQuantMix(const k1::ModelWeights& w) {
  const auto v_high = Count(w, GgmlType::kQ6_K, &k1::LayerWeights::attn_v);
  const auto down_high =
      Count(w, GgmlType::kQ6_K, &k1::LayerWeights::down_exps);
  const auto shexp_high =
      Count(w, GgmlType::kQ6_K, &k1::LayerWeights::down_shexp);
  Require(v_high == kHighQuantLayers && down_high == kHighQuantLayers &&
              shexp_high == kHighQuantLayers,
          "half of the layers keep Q6_K on the value and down projections");
  for (const auto& layer : w.layers) {
    const bool v_is_high = layer.attn_v.type == GgmlType::kQ6_K;
    Require(v_is_high == (layer.down_exps.type == GgmlType::kQ6_K) &&
                v_is_high == (layer.down_shexp.type == GgmlType::kQ6_K),
            "the value and down projections of one layer share a format");
  }
  // One expert of the 2560 x 512 gate and of the 512 x 2560 down projection.
  // These strides are what the grouped kernels address, so they are pinned.
  Require(w.layers[6].gate_exps.ExpertBytes() == 737280,
          "one Q4_K expert of the gate projection");
  Require(w.layers[0].down_exps.ExpertBytes() == 1075200,
          "one Q6_K expert of the down projection");
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_KOLIBRI1_MODEL");
  if (model == nullptr || *model == '\0') {
    std::cout << "SKIP: GUFO_KOLIBRI1_MODEL is not set\n";
    return 77;
  }
  return kolibri1_test::Run([&] {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Require(reader != nullptr, error);
    const auto weights = k1::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), "artifact rejected: " + error);
    // The renderer is compiled against one revision of the template; a file
    // carrying a different one must be refused at load, not silently rendered.
    std::string template_error;
    Require(k1::ChatTemplate::ValidateGgufTemplate(*reader, &template_error),
            "chat template: " + template_error);
    CheckConfig(weights->config);
    CheckLayout(*weights);
    CheckQuantMix(*weights);
    std::cout << "qualified chat template " << k1::ChatTemplate::TemplateId()
              << "\n";
    std::cout << "bound " << weights->config.num_layers << " layers, "
              << weights->config.num_experts << " experts, vocab "
              << weights->VocabSize() << "\n";
  });
}
