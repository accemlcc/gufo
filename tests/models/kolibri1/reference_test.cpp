#include "src/models/kolibri1/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kolibri1/weights.hpp"

#ifndef GUFO_KOLIBRI1_FIXTURES
#define GUFO_KOLIBRI1_FIXTURES "."
#endif

namespace {

namespace kolibri = gufo::models::kolibri1;

std::string Fixture(const char* name) {
  return std::string(GUFO_KOLIBRI1_FIXTURES) + "/" + name;
}

struct Goldens {
  std::vector<std::int32_t> tokens;
  std::vector<float> logits;
  std::size_t vocab{};
};

Goldens LoadGoldens() {
  FILE* file = std::fopen(Fixture("kolibri1_logits.bin").c_str(), "rb");
  if (file == nullptr) {
    throw std::runtime_error(
        "cannot read kolibri1_logits.bin; regenerate it "
        "with "
        "tests/models/kolibri1/make_reference_fixture.py");
  }
  const auto Read = [&](void* data, std::size_t count) {
    if (std::fread(data, 1, count, file) != count) {
      std::fclose(file);
      throw std::runtime_error("truncated reference fixture");
    }
  };
  Goldens goldens;
  std::uint32_t tokens = 0;
  std::uint32_t vocab = 0;
  Read(&tokens, sizeof(tokens));
  Read(&vocab, sizeof(vocab));
  goldens.vocab = vocab;
  goldens.tokens.resize(tokens);
  goldens.logits.resize(static_cast<std::size_t>(tokens) * goldens.vocab);
  Read(goldens.tokens.data(), goldens.tokens.size() * sizeof(std::int32_t));
  Read(goldens.logits.data(), goldens.logits.size() * sizeof(float));
  std::fclose(file);
  return goldens;
}

/// The reader owns the mapping the weight payloads point into, so it has to
/// outlive the bound view.
struct FixtureModel {
  std::unique_ptr<gufo::core::GgufReader> reader;
  kolibri::ModelWeights weights;
};

FixtureModel LoadModel() {
  std::string error;
  std::unique_ptr<gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(Fixture("kolibri1_tiny.gguf"), &error);
  if (reader == nullptr) {
    throw std::runtime_error("cannot map kolibri1_tiny.gguf: " + error +
                             "; regenerate it with "
                             "tests/models/kolibri1/make_reference_fixture.py");
  }
  std::optional<kolibri::ModelWeights> weights =
      kolibri::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    throw std::runtime_error("Bind rejected the reference fixture: " + error);
  }
  return {std::move(reader), std::move(*weights)};
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0F;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

std::size_t Argmax(const float* row, std::size_t size) {
  return static_cast<std::size_t>(std::max_element(row, row + size) - row);
}

/// A layer's residual for the last row of a traced call. The trace is laid out
/// [layer][row][hidden].
std::vector<float> LastRow(const std::vector<float>& trace, std::size_t layer,
                           std::size_t rows, std::size_t hidden) {
  const std::size_t stride = rows * hidden;
  return {trace.begin() + layer * stride + (rows - 1) * hidden,
          trace.begin() + layer * stride + rows * hidden};
}

}  // namespace

int main() {
  try {
    const Goldens goldens = LoadGoldens();
    const FixtureModel model = LoadModel();
    const std::size_t vocab = goldens.vocab;
    const std::size_t hidden = model.weights.config.hidden_size;

    // 1. The oracle must reproduce the independent NumPy forward: same layout,
    //    same norms, same routing, same window.
    kolibri::Reference reference(model.weights);
    std::vector<float> logits;
    std::vector<float> normed;
    reference.Forward(goldens.tokens, &logits, &normed);
    if (logits.size() != goldens.logits.size()) {
      std::fprintf(stderr, "FAIL: logits shape %zu != %zu\n", logits.size(),
                   goldens.logits.size());
      return 1;
    }
    for (std::size_t row = 0; row * vocab < logits.size(); ++row) {
      if (Argmax(&logits[row * vocab], vocab) !=
          Argmax(&goldens.logits[row * vocab], vocab)) {
        std::fprintf(stderr, "FAIL: row %zu picks another token\n", row);
        return 1;
      }
    }
    const float diff = MaxAbsDiff(logits, goldens.logits);
    if (diff > 1e-4F) {
      std::fprintf(stderr, "FAIL: logits differ by %.3e\n", diff);
      return 1;
    }
    std::printf("reference: logits match the NumPy oracle (max %.3e)\n", diff);

    // 2. The same tokens fed in two calls must give the same result: the cache
    //    stores rotated keys and the window has to keep working across calls.
    reference.Reset();
    const std::vector<std::int32_t> head(goldens.tokens.begin(),
                                         goldens.tokens.begin() + 2);
    const std::vector<std::int32_t> tail(goldens.tokens.begin() + 2,
                                         goldens.tokens.end());
    // Each call writes the rows of its own tokens, so the two chunks have to
    // be concatenated before the comparison.
    std::vector<float> head_logits;
    std::vector<float> tail_logits;
    reference.Forward(head, &head_logits, nullptr);
    reference.Forward(tail, &tail_logits, nullptr);
    std::vector<float> chunk_logits = head_logits;
    chunk_logits.insert(chunk_logits.end(), tail_logits.begin(),
                        tail_logits.end());
    if (chunk_logits.size() != goldens.logits.size()) {
      std::fprintf(stderr,
                   "FAIL: chunked forward wrote %zu values, expected %zu\n",
                   chunk_logits.size(), goldens.logits.size());
      return 1;
    }
    const float chunked = MaxAbsDiff(chunk_logits, goldens.logits);
    if (chunked > 1e-4F) {
      std::fprintf(stderr, "FAIL: chunked forward differs by %.3e\n", chunked);
      return 1;
    }
    std::printf("reference: chunked forward matches (max %.3e)\n", chunked);

    // 3. Window semantics. Two prompts share their last WINDOW tokens and
    //    differ only further back: the sliding layer's last row must not move,
    //    the global layer's must.
    reference.SetTrace(true);
    const std::vector<std::int32_t> suffix = {2, 0, 4, 6, 1};
    const std::vector<std::int32_t> prefix_a = {3, 5, 1, 6, 0};
    const std::vector<std::int32_t> prefix_b = {4, 5, 6, 6, 0};
    const std::size_t rows = prefix_a.size() + suffix.size();

    const auto Trace = [&](const std::vector<std::int32_t>& prefix,
                           const std::vector<std::int32_t>& tail) {
      reference.Reset();
      std::vector<std::int32_t> tokens;
      tokens.insert(tokens.end(), prefix.begin(), prefix.end());
      tokens.insert(tokens.end(), tail.begin(), tail.end());
      std::vector<float> unused;
      reference.Forward(tokens, &unused, nullptr);
      return reference.LayerTrace();
    };
    const std::vector<float> a = Trace(prefix_a, suffix);
    const std::vector<float> b = Trace(prefix_b, suffix);

    const float sliding_delta =
        MaxAbsDiff(LastRow(a, 0, rows, hidden), LastRow(b, 0, rows, hidden));
    if (sliding_delta > 0.0F) {
      std::fprintf(stderr,
                   "FAIL: sliding layer saw keys outside its window (%.3e)\n",
                   sliding_delta);
      return 1;
    }
    const float global_delta =
        MaxAbsDiff(LastRow(a, 2, rows, hidden), LastRow(b, 2, rows, hidden));
    if (!(global_delta > 1e-6F)) {
      std::fprintf(stderr, "FAIL: global layer ignored the earlier tokens\n");
      return 1;
    }
    std::printf(
        "reference: sliding layer blind behind the window, global "
        "layer still uses those keys (delta %.3e)\n",
        global_delta);

    // The comparison above only means something if the sliding layer does
    // react to keys inside its window.
    std::vector<std::int32_t> changed_suffix = suffix;
    changed_suffix.back() = 3;
    const std::vector<float> c = Trace(prefix_a, changed_suffix);
    if (!(MaxAbsDiff(LastRow(a, 0, rows, hidden), LastRow(c, 0, rows, hidden)) >
          1e-6F)) {
      std::fprintf(stderr, "FAIL: sliding layer ignored a key in its window\n");
      return 1;
    }
    std::printf("reference: sliding layer reacts to keys inside the window\n");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
  std::printf("kolibri1_reference_test: OK\n");
  return 0;
}
