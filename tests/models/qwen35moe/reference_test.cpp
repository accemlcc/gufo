// The scalar oracle against the NumPy fixture of the same tiny artifact;
// both must implement the graph independently, so the check is slow in
// debug builds and keeps -O2.
#ifndef GUFO_QWEN35MOE_FIXTURES
#define GUFO_QWEN35MOE_FIXTURES "."
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35moe/reference.hpp"
#include "src/models/qwen35moe/weights.hpp"
#include "tests/models/qwen35moe/check.hpp"

namespace q35 = gufo::models::qwen35moe;
using qwen35moe_test::Require;

namespace {

std::string Fixture(const std::string& name) {
  return std::string(GUFO_QWEN35MOE_FIXTURES) + "/" + name;
}

struct FixtureData {
  std::vector<std::int32_t> tokens;
  std::uint32_t vocab = 0;
  std::vector<float> logits;
};

FixtureData ReadLogits() {
  FixtureData out;
  FILE* file = std::fopen(Fixture("qwen35moe_logits.bin").c_str(), "rb");
  Require(file != nullptr, "cannot open qwen35moe_logits.bin");
  std::uint32_t n = 0;
  Require(std::fread(&n, 4, 1, file) == 1, "short fixture header");
  Require(std::fread(&out.vocab, 4, 1, file) == 1, "short fixture header");
  out.tokens.resize(n);
  Require(std::fread(out.tokens.data(), 4, n, file) == n, "short tokens");
  out.logits.resize(static_cast<std::size_t>(n) * out.vocab);
  Require(std::fread(out.logits.data(), 4, out.logits.size(), file) ==
              out.logits.size(),
          "short logits");
  std::fclose(file);
  return out;
}

void ReferenceMatchesFixture() {
  const FixtureData fx = ReadLogits();
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(
      Fixture("qwen35moe_tiny.gguf"), &error);
  Require(reader != nullptr, "cannot open fixture artifact: " + error);
  const auto weights = q35::ModelWeights::LoadFromGguf(*reader, &error);
  Require(weights.has_value(), "fixture artifact rejected: " + error);
  q35::Reference ref(*weights);
  std::vector<float> logits;
  ref.Forward(fx.tokens, &logits);
  Require(logits.size() == fx.logits.size(), "logit row count");
  double max_abs = 0.0;
  double mean_abs = 0.0;
  std::size_t ref_argmax = 0;
  std::size_t fx_argmax = 0;
  const std::size_t last = fx.tokens.size() - 1;
  for (std::size_t i = 0; i < fx.logits.size(); ++i) {
    const double diff =
        std::fabs(static_cast<double>(logits[i]) - fx.logits[i]);
    max_abs = std::max(max_abs, diff);
    mean_abs += diff;
  }
  mean_abs /= fx.logits.size();
  for (std::uint32_t v = 0; v < fx.vocab; ++v) {
    if (logits[last * fx.vocab + v] > logits[last * fx.vocab + ref_argmax]) {
      ref_argmax = v;
    }
    if (fx.logits[last * fx.vocab + v] >
        fx.logits[last * fx.vocab + fx_argmax]) {
      fx_argmax = v;
    }
  }
  Require(ref_argmax == fx_argmax, "frontier argmax agrees");
  Require(max_abs < 1e-3, "max_abs within float32 rounding");
  Require(mean_abs < 1e-4, "mean_abs within float32 rounding");
}

void ChunkInvariant() {
  const FixtureData fx = ReadLogits();
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(
      Fixture("qwen35moe_tiny.gguf"), &error);
  Require(reader != nullptr, "cannot open fixture artifact: " + error);
  const auto weights = q35::ModelWeights::LoadFromGguf(*reader, &error);
  Require(weights.has_value(), "fixture artifact rejected: " + error);
  q35::Reference whole(*weights);
  std::vector<float> all;
  whole.Forward(fx.tokens, &all);
  q35::Reference split(*weights);
  const std::size_t half = fx.tokens.size() / 2;
  std::vector<float> first;
  std::vector<std::int32_t> head(fx.tokens.begin(),
                                 fx.tokens.begin() + half);
  std::vector<std::int32_t> tail(fx.tokens.begin() + half, fx.tokens.end());
  std::vector<float> ignored;
  split.Forward(head, &ignored);
  split.Forward(tail, &first);
  Require(first.size() == tail.size() * fx.vocab, "tail logit rows");
  for (std::size_t r = 0; r < tail.size(); ++r) {
    for (std::uint32_t v = 0; v < fx.vocab; ++v) {
      Require(first[r * fx.vocab + v] ==
                  all[(half + r) * fx.vocab + v],
              "split append is bit-identical");
    }
  }
}

}  // namespace

int main() {
  return qwen35moe_test::Run([] {
    ReferenceMatchesFixture();
    ChunkInvariant();
  });
}
