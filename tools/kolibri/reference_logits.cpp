// Dumps Kolibri 1 logits from the scalar reference for cross-checking.
//
// The reference is the only Kolibri 1 forward pass in this repository that runs
// without HIP, so it is what a reference implementation - patched llama.cpp -
// gets compared against. Input is token ids, never text: a parity run has to
// pin ids so that a tokenizer difference cannot masquerade as a graph
// difference. One prompt per line; the logits of each prompt's last position
// are written out, which is the row a fresh-context reference run produces.
//
// Usage: reference_logits <model.gguf> <out.json> [prompts-file]
// With no prompts-file, the prompt ids come from the remaining arguments.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kolibri1/reference.hpp"
#include "src/models/kolibri1/weights.hpp"

namespace {

bool Fail(const std::string& message) {
  std::fprintf(stderr, "reference_logits: %s\n", message.c_str());
  return false;
}

/// Parses one prompt line of whitespace-separated ids; blank lines are skipped.
bool ParsePrompt(const std::string& line, std::vector<std::int32_t>* prompt,
                 std::string* error) {
  std::istringstream stream(line);
  std::string field;
  while (stream >> field) {
    char* end = nullptr;
    const long id = std::strtol(field.c_str(), &end, 10);
    if (end == field.c_str() || *end != '\0' || id < 0 || id > 0x7FFFFFFFL) {
      *error = "not a token id: '" + field + "'";
      return false;
    }
    prompt->push_back(static_cast<std::int32_t>(id));
  }
  return true;
}

std::vector<std::vector<std::int32_t>> ReadPrompts(int argc, char** argv,
                                                   std::string* error) {
  std::vector<std::vector<std::int32_t>> prompts;
  if (argc > 3) {
    std::ifstream file(argv[3]);
    if (!file) {
      *error = std::string("cannot read prompts file ") + argv[3];
      return prompts;
    }
    for (std::string line; std::getline(file, line);) {
      std::vector<std::int32_t> prompt;
      if (!ParsePrompt(line, &prompt, error)) {
        prompts.clear();
        return prompts;
      }
      if (!prompt.empty()) {
        prompts.push_back(std::move(prompt));
      }
    }
    return prompts;
  }
  std::vector<std::int32_t> prompt;
  std::ostringstream joined;
  for (int arg = 3; arg < argc; ++arg) {
    if (arg != 3) {
      joined << ' ';
    }
    joined << argv[arg];
  }
  if (!ParsePrompt(joined.str(), &prompt, error)) {
    return prompts;
  }
  prompts.push_back(std::move(prompt));
  return prompts;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(
        stderr,
        "usage: %s <model.gguf> <out.json> [prompts-file | token-id ...]\n",
        argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string out_path = argv[2];

  std::string error;
  const auto prompts = ReadPrompts(argc, argv, &error);
  if (prompts.empty()) {
    return Fail(error.empty() ? "no prompts" : error) ? 0 : 1;
  }

  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    return Fail(error) ? 0 : 1;
  }
  const auto weights =
      gufo::models::kolibri1::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    return Fail("artifact rejected: " + error) ? 0 : 1;
  }
  // The oracle is deliberately exact: dequantised weights read straight from
  // the artifact, F64 accumulators, F32 cache. Measured against the patched
  // llama.cpp build it differs by about 5e-4 relative per projection already in
  // layer 0, and that grows to a logit spread of ~3 over the 50 layers, because
  // llama's CPU matmul feeds its quantised weights an int8 copy of the
  // activation. None of that is a semantics difference - on an all-F32 artifact
  // the two agree per operator to ~1e-6 - so the oracle stays exact and the
  // kernels are validated against it, not against llama's CPU rounding.
  gufo::models::kolibri1::Reference reference(*weights);
  const char* trace_dir = std::getenv("GUFO_KOLIBRI_TRACE");
  if (trace_dir != nullptr && *trace_dir != '\0') {
    reference.SetTrace(true);
  }
  // GUFO_KOLIBRI_STAGES=dir additionally dumps the named intermediates the
  // reference graph traces, so a divergence can be pinned to one operator
  // rather than to one layer. It reflects the last prompt, like the layer
  // trace does.
  const char* stage_dir = std::getenv("GUFO_KOLIBRI_STAGES");
  if (stage_dir != nullptr && *stage_dir != '\0') {
    reference.SetStageTrace(true);
  }
  const std::size_t vocab = weights->VocabSize();

  std::ofstream out(out_path);
  if (!out) {
    return Fail("cannot write " + out_path) ? 0 : 1;
  }
  out << "{\"vocab_size\": " << vocab << ", \"prompts\": [";
  std::vector<float> logits;
  for (std::size_t index = 0; index < prompts.size(); ++index) {
    // Each prompt runs in a fresh context, like a reference run that reloads.
    reference.Reset();
    logits.clear();
    reference.Forward(prompts[index], &logits);
    if (!out.good()) {
      return Fail("write failed") ? 0 : 1;
    }
    if (index != 0) {
      out << ", ";
    }
    out << "{\"tokens\": [";
    for (std::size_t i = 0; i < prompts[index].size(); ++i) {
      if (i != 0) {
        out << ',';
      }
      out << prompts[index][i];
    }
    // The last row first: `logits` holds [rows][vocab], so row 0 is what a
    // bare `logits[token]` would print - and that is a different position than
    // the single-row dump the comparison used to line up against.
    const std::size_t last = prompts[index].size() - 1;
    out << "], \"logits\": [";
    for (std::size_t token = 0; token < vocab; ++token) {
      if (token != 0) {
        out << ',';
      }
      out << logits[last * vocab + token];
    }
    out << "]";
    // All rows, so a mismatch can be attributed to a position instead of only
    // to the last one - a causal or positional bug hides in a single row.
    out << ", \"rows\": [";
    for (std::size_t row = 0; row < prompts[index].size(); ++row) {
      if (row != 0) {
        out << ", ";
      }
      out << "[";
      for (std::size_t token = 0; token < vocab; ++token) {
        if (token != 0) {
          out << ',';
        }
        out << logits[row * vocab + token];
      }
      out << "]";
    }
    out << "]}";
    out.flush();
    std::fprintf(stderr, "prompt %zu (%zu tokens) done\n", index,
                 prompts[index].size());
  }
  out << "]}\n";
  if (!out.good()) {
    return Fail("write failed") ? 0 : 1;
  }

  if (trace_dir != nullptr && *trace_dir != '\0') {
    const std::vector<float>& trace = reference.LayerTrace();
    const std::string path = std::string(trace_dir) + "/l_out.json";
    std::ofstream layer_out(path);
    if (!layer_out) {
      return Fail("cannot write " + path) ? 0 : 1;
    }
    const std::size_t rows = prompts.back().size();
    layer_out << "{\"layers\": " << weights->config.num_layers
              << ", \"rows\": " << rows << ", \"hidden\": " << weights->config.hidden_size
              << ", \"data\": [";
    for (std::size_t i = 0; i < trace.size(); ++i) {
      if (i != 0) {
        layer_out << ',';
      }
      layer_out << trace[i];
    }
    layer_out << "]}\n";
    std::fprintf(stderr, "wrote %s\n", path.c_str());
  }
  if (stage_dir != nullptr && *stage_dir != '\0') {
    // Same shape as the llama.cpp side writes, so one comparer can diff the
    // two by name. Values are flat row-major [row][feature] like every tensor
    // here except the weights.
    const std::map<std::string, std::vector<float>>& stages =
        reference.StageTrace();
    const std::string path = std::string(stage_dir) + "/stages.json";
    std::ofstream stage_out(path);
    if (!stage_out) {
      return Fail("cannot write " + path) ? 0 : 1;
    }
    stage_out.precision(9);
    stage_out << "{\"rows\": " << prompts.back().size()
              << ", \"hidden\": " << weights->config.hidden_size
              << ", \"stages\": {";
    bool first = true;
    for (const auto& entry : stages) {
      if (!first) {
        stage_out << ", ";
      }
      first = false;
      stage_out << '"' << entry.first << "\": [";
      for (std::size_t i = 0; i < entry.second.size(); ++i) {
        if (i != 0) {
          stage_out << ',';
        }
        stage_out << entry.second[i];
      }
      stage_out << ']';
    }
    stage_out << "}}\n";
    std::fprintf(stderr, "wrote %s (%zu stages)\n", path.c_str(),
                 stages.size());
  }
  return 0;
}
