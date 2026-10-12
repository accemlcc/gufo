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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// Residual stream after every block, collected from the graph callback. The
/// model calls its per-layer output "l_out-<index>", which is what the callback
/// name carries.
struct Trace {
  std::vector<std::vector<float>> layers;
  uint32_t width{0};
};

bool TraceCallback(struct ggml_tensor* t, bool ask, void* user) {
  auto* trace = static_cast<Trace*>(user);
  const std::string name = ggml_get_name(t);
  constexpr std::string_view kPrefix = "l_out-";
  if (name.rfind(kPrefix, 0) != 0 || t->type != GGML_TYPE_F32) {
    return ask ? false : true;
  }
  if (ask) {
    return true;
  }
  const auto layer = static_cast<std::size_t>(std::stoul(name.substr(kPrefix.size())));
  if (trace->layers.size() <= layer) {
    trace->layers.resize(layer + 1);
  }
  trace->width = static_cast<uint32_t>(t->ne[0]);
  std::vector<float> data(ggml_nelements(t));
  ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
  auto& dst = trace->layers[layer];
  dst.insert(dst.end(), data.begin(), data.end());
  return true;
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
  Trace trace;
  if (trace_path != nullptr && *trace_path != '\0') {
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
    for (auto& layer : trace.layers) {
      layer.clear();
    }
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
  if (trace_path != nullptr && *trace_path != '\0') {
    // Same JSON shape as reference_logits' GUFO_KOLIBRI_TRACE output: the last
    // prompt's residual stream, layer by layer.
    const std::size_t rows = prompts.back().size();
    std::ofstream layer_out(trace_path);
    if (!layer_out) {
      Die(std::string("cannot write ") + trace_path);
    }
    layer_out << "{\"layers\": " << trace.layers.size() << ", \"rows\": " << rows
              << ", \"hidden\": " << trace.width << ", \"data\": [";
    for (std::size_t i = 0; i < trace.layers.size(); ++i) {
      for (std::size_t j = 0; j < trace.layers[i].size(); ++j) {
        layer_out << (i == 0 && j == 0 ? "" : ",") << trace.layers[i][j];
      }
    }
    layer_out << "]}\n";
    std::fprintf(stderr, "wrote %s\n", trace_path);
  }
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
