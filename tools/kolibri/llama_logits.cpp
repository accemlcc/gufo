// Teacher-forced logits from a reference llama.cpp build, for Kolibri 1 parity.
//
// Emitted JSON matches tools/kolibri/reference_logits.cpp field for field, so
// compare_logits.py can line the two engines up without knowing which is which.
// The reference graph runs on the CPU here: this is a semantics check on a
// small artifact, not a performance measurement, and CPU keeps it out of the
// way of the resident engine. Each prompt starts from a cleared cache, which is
// what a fresh sequence in llama.cpp means for an iSWA model.
//
// Usage: llama_logits <model.gguf> <out.json> [prompts-file]
// Build: tools/kolibri/build_llama_logits.sh
#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

/// Named intermediates collected from the graph callback. The oracle in
/// src/models/kolibri1 records the same points under the same names, so a
/// divergence can be attributed to one operator instead of to one layer. The
/// model suffixes per-layer tensors with "-<index>" at build time and appends
/// "-<il>" when the callback fires, so the keys here are already comparable.
struct Trace {
  uint32_t width{0};
  std::map<std::string, std::vector<float>> stages;
};

constexpr std::string_view kStageNames[] = {
    "attn_norm",  "Qcur_normed", "Qcur_rope",   "Kcur_normed",
    "Kcur_rope",  "Vcur",        "attn_out",     "attn_post_norm",
    "ffn_inp",    "ffn_norm",    "ffn_moe_out",  "ffn_shexp",
    "ffn_out",    "ffn_post_norm", "l_out",      "result_norm",
};

/// Strips a trailing "-<digits>" layer suffix.
bool SplitLayerSuffix(const std::string& name, std::string_view* base) {
  const std::size_t dash = name.rfind('-');
  if (dash == std::string::npos || dash + 1 >= name.size()) {
    *base = name;
    return false;
  }
  for (std::size_t i = dash + 1; i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') {
      *base = name;
      return false;
    }
  }
  *base = std::string_view(name).substr(0, dash);
  return true;
}

bool TraceCallback(struct ggml_tensor* t, bool ask, void* user) {
  auto* trace = static_cast<Trace*>(user);
  if (t->type != GGML_TYPE_F32) {
    return ask ? false : true;
  }
  const std::string name = ggml_get_name(t);
  std::string_view base;
  SplitLayerSuffix(name, &base);
  for (const std::string_view wanted : kStageNames) {
    if (wanted != base) {
      continue;
    }
    if (ask) {
      return true;
    }
    std::vector<float>& dst = trace->stages[name];
    dst.resize(static_cast<std::size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    trace->width = static_cast<uint32_t>(dst.size());
    return true;
  }
  return ask ? false : true;
}

[[noreturn]] void Die(const std::string& message) {
  std::fprintf(stderr, "llama_logits: %s\n", message.c_str());
  std::exit(1);
}

bool ParsePrompt(const std::string& line, std::vector<llama_token>* prompt,
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
    prompt->push_back(static_cast<llama_token>(id));
  }
  return true;
}

std::vector<std::vector<llama_token>> ReadPrompts(int argc, char** argv,
                                                  std::string* error) {
  std::vector<std::vector<llama_token>> prompts;
  std::vector<std::string> lines;
  if (argc > 3) {
    std::ifstream file(argv[3]);
    if (!file) {
      *error = std::string("cannot read prompts file ") + argv[3];
      return prompts;
    }
    for (std::string line; std::getline(file, line);) {
      lines.push_back(line);
    }
  } else {
    std::ostringstream joined;
    for (int arg = 3; arg < argc; ++arg) {
      if (arg != 3) {
        joined << ' ';
      }
      joined << argv[arg];
    }
    lines.push_back(joined.str());
  }
  for (const auto& line : lines) {
    std::vector<llama_token> prompt;
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
  auto prompts = ReadPrompts(argc, argv, &error);
  if (prompts.empty()) {
    Die(error.empty() ? "no prompts" : error);
  }

  llama_backend_init();
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;  // semantics check, not a speed run
  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  if (model == nullptr) {
    Die("cannot load " + model_path);
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);

  auto context_params = llama_context_default_params();
  std::size_t longest = 1;
  for (const auto& prompt : prompts) {
    longest = std::max(longest, prompt.size());
  }
  context_params.n_ctx = static_cast<uint32_t>(
      std::max<std::size_t>(longest + 8, llama_model_n_ctx_train(model) > 0
                                ? 64
                                : 64));
  context_params.n_batch = static_cast<uint32_t>(longest);
  context_params.n_ubatch = context_params.n_batch;
  context_params.n_seq_max = 1;
  // Two knobs so a parity run can state which numeric path it measured.
  // GUFO_LLAMA_KV=f32 pins the cache to f32 (the default f16 keeps keys and
  // values at an 11-bit mantissa) and GUFO_LLAMA_FLASH=0 disables flash
  // attention. Both were added while attributing the residual difference to
  // the oracle; measured on the f32 artifact, neither of them is the cause —
  // the numbers are recorded in tools/kolibri/run_parity.sh.
  if (const char* kv = std::getenv("GUFO_LLAMA_KV");
      kv != nullptr && std::string(kv) == "f32") {
    context_params.type_k = GGML_TYPE_F32;
    context_params.type_v = GGML_TYPE_F32;
  }
  if (const char* flash = std::getenv("GUFO_LLAMA_FLASH");
      flash != nullptr && std::string(flash) == "0") {
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  }
  std::fprintf(stderr, "kv cache: K=%s V=%s\n", ggml_type_name(context_params.type_k),
               ggml_type_name(context_params.type_v));
  const char* trace_path = std::getenv("GUFO_LLAMA_TRACE");
  const char* stages_path = std::getenv("GUFO_LLAMA_STAGES");
  Trace trace;
  if ((trace_path != nullptr && *trace_path != '\0') ||
      (stages_path != nullptr && *stages_path != '\0')) {
    context_params.cb_eval = TraceCallback;
    context_params.cb_eval_user_data = &trace;
  }
  llama_context* ctx = llama_init_from_model(model, context_params);
  if (ctx == nullptr) {
    Die("cannot create a context");
  }

  std::ofstream out(out_path);
  if (!out) {
    Die("cannot write " + out_path);
  }
  out << "{\"vocab_size\": " << n_vocab << ", \"prompts\": [";
  for (std::size_t index = 0; index < prompts.size(); ++index) {
    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    // Keep only the latest prompt in the trace: the comparison runs one prompt
    // at a time.
    trace.stages.clear();
    // A manual batch so every position asks for logits; llama_batch_get_one
    // only flags the last one, which would hide a position-dependent mismatch.
    llama_batch batch =
        llama_batch_init(static_cast<int32_t>(prompts[index].size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(prompts[index].size());
    for (std::size_t row = 0; row < prompts[index].size(); ++row) {
      batch.token[row] = prompts[index][row];
      batch.pos[row] = static_cast<llama_pos>(row);
      batch.n_seq_id[row] = 1;
      batch.seq_id[row][0] = 0;
      batch.logits[row] = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
      llama_batch_free(batch);
      Die("decode failed at prompt " + std::to_string(index));
    }
    // Row pointers by batch index. The negative index form of
    // llama_get_logits_ith does not mean "last position" for a batch with
    // several flagged outputs, so the last row is addressed by its index.
    std::vector<const float*> rows;
    for (std::size_t row = 0; row < prompts[index].size(); ++row) {
      const float* row_logits =
          llama_get_logits_ith(ctx, static_cast<int32_t>(row));
      if (row_logits == nullptr) {
        llama_batch_free(batch);
        Die("missing logits for row " + std::to_string(row));
      }
      rows.push_back(row_logits);
    }
    const float* logits = rows.back();
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
    out << "], \"logits\": [";
    for (int32_t token = 0; token < n_vocab; ++token) {
      if (token != 0) {
        out << ',';
      }
      out << logits[token];
    }
    out << "]";
    out << ", \"rows\": [";
    for (std::size_t row = 0; row < rows.size(); ++row) {
      const float* row_logits = rows[row];
      if (row != 0) {
        out << ", ";
      }
      out << "[";
      for (int32_t token = 0; token < n_vocab; ++token) {
        if (token != 0) {
          out << ',';
        }
        out << row_logits[token];
      }
      out << "]";
    }
    out << "]}";
    llama_batch_free(batch);
    out.flush();
    std::fprintf(stderr, "prompt %zu (%zu tokens) done\n", index,
                 prompts[index].size());
  }
  out << "]}\n";
  if (!out.good()) {
    Die("write failed");
  }
  const std::size_t rows = prompts.back().size();
  if (trace_path != nullptr && *trace_path != '\0') {
    // Same JSON shape as reference_logits' GUFO_KOLIBRI_TRACE output: the last
    // prompt's residual stream, layer by layer.
    std::ofstream layer_out(trace_path);
    if (!layer_out) {
      Die(std::string("cannot write ") + trace_path);
    }
    std::size_t layers = 0;
    while (trace.stages.count("l_out-" + std::to_string(layers)) != 0) {
      ++layers;
    }
    const std::size_t hidden =
        layers == 0
            ? 0
            : trace.stages.at("l_out-0").size() / std::max<std::size_t>(rows, 1);
    layer_out << "{\"layers\": " << layers << ", \"rows\": " << rows
              << ", \"hidden\": " << hidden << ", \"data\": [";
    bool first = true;
    for (std::size_t layer = 0; layer < layers; ++layer) {
      for (const float value : trace.stages.at("l_out-" + std::to_string(layer))) {
        layer_out << (first ? "" : ",") << value;
        first = false;
      }
    }
    layer_out << "]}\n";
    std::fprintf(stderr, "wrote %s\n", trace_path);
  }
  if (stages_path != nullptr && *stages_path != '\0') {
    // Field-for-field the same shape as GUFO_KOLIBRI_STAGES on the oracle, so
    // compare_stages.py can diff the two engines operator by operator.
    std::ofstream stages_out(stages_path);
    if (!stages_out) {
      Die(std::string("cannot write ") + stages_path);
    }
    stages_out.precision(9);
    stages_out << "{\"rows\": " << rows << ", \"stages\": {";
    bool first = true;
    for (const auto& entry : trace.stages) {
      if (!first) {
        stages_out << ", ";
      }
      first = false;
      stages_out << '"' << entry.first << "\": [";
      for (std::size_t i = 0; i < entry.second.size(); ++i) {
        stages_out << (i == 0 ? "" : ",") << entry.second[i];
      }
      stages_out << ']';
    }
    stages_out << "}}\n";
    std::fprintf(stderr, "wrote %s (%zu stages)\n", stages_path,
                 trace.stages.size());
  }
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
