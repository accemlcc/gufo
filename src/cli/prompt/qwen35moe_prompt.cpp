// `gufo prompt` and `gufo chat` for Ornith-1.5 (Qwen3.5-MoE) targets: render
// the file's own chat template, tokenize without a BOS marker (the metadata
// says add_bos is false, and the parity harness compares raw id streams),
// prefill in chunks and sample to the stop token.
#include "src/cli/prompt/qwen35moe_prompt.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen35moe/chat_template.hpp"
#include "src/models/qwen35moe/engine.hpp"

namespace gufo::cli {
namespace {

namespace q35 = gufo::models::qwen35moe;

/// The context one conversation may use. The 35B artifact allows far more,
/// but 8k rows of KV cache stay well under a gigabyte against 25 GiB of
/// weights, which is what an interactive run wants.
constexpr std::uint32_t kQwen35MoeContext = 8192;

double Elapsed(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

[[nodiscard]] std::unique_ptr<q35::Model> LoadModel(
    const PromptOptions& opt,
    std::chrono::steady_clock::time_point load_start) {
  std::string error;
  auto model = q35::Model::Load(opt.model_path,
                                q35::ModelOptions{
                                    .max_context = kQwen35MoeContext,
                                    .prefill_chunk = 16,
                                },
                                &error);
  if (!model) {
    std::cerr << "Ornith-1.5 load failed: " << error << '\n';
    return nullptr;
  }
  if (opt.verbose) {
    std::cout << "Model loaded in " << Elapsed(load_start) << " s; "
              << model->resident_bytes() / (1024U * 1024U)
              << " MiB of weights on the device\n";
  }
  return model;
}

/// Tokenizes a rendered prompt. Control tokens in the template text are matched
/// as specials, which is what makes the role markers single tokens. No BOS
/// marker: the artifact metadata says add_bos is false.
[[nodiscard]] std::vector<q35::TokenId> TokenizePrompt(
    const q35::Model& model, const std::string& text) {
  tokenization::TokenizerOptions options;
  options.add_bos = false;
  options.parse_special_tokens = true;
  std::vector<tokenization::QwenTokenizer::TokenId> ids =
      model.tokenizer().Encode(text, options);
  std::vector<q35::TokenId> out(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    out[i] = static_cast<q35::TokenId>(ids[i]);
  }
  return out;
}

/// Prefills `prompt` and generates until the model stops or `opt.max_tokens`
/// are out. `reply` collects the text of the turn when the caller keeps
/// history.
int Generate(const PromptOptions& opt, const q35::Model& model,
             q35::Session& session, std::span<const q35::TokenId> prompt,
             std::string* reply) {
  if (prompt.empty()) {
    std::cerr << "Ornith-1.5 prompt produced no tokens\n";
    return 1;
  }
  if (prompt.size() + opt.max_tokens > model.max_context()) {
    std::cerr << "Ornith-1.5 prompt and output exceed the "
              << model.max_context() << "-token CLI context\n";
    return 1;
  }
  const auto prefill_start = std::chrono::steady_clock::now();
  // A turn is a whole state: whatever this session answered before is dropped.
  session.Reset();
  session.Append(prompt);
  const double prefill_seconds = Elapsed(prefill_start);
  if (opt.verbose) {
    std::cout << "[Engine]: Ornith-1.5 ROCm (gfx1151)\n"
              << "Prompt tokens: " << prompt.size() << '\n'
              << "--- Generation Output ---\n";
  }
  // The sampler works in unsigned token ids and the engine in signed ones, so
  // the prompt is handed over as a copy rather than a view.
  std::vector<sampling::TokenId> history;
  history.reserve(prompt.size());
  for (const q35::TokenId token : prompt) {
    history.push_back(static_cast<sampling::TokenId>(token));
  }
  sampling::SamplerState sampler(opt.sampling, history);
  const auto decode_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  for (; generated < opt.max_tokens; ++generated) {
    const q35::TokenId token = static_cast<q35::TokenId>(
        sampler.Sample(session.logits()));
    sampler.Accept(static_cast<sampling::TokenId>(token));
    if (model.tokenizer().IsStopToken(
            static_cast<tokenization::QwenTokenizer::TokenId>(token))) {
      break;
    }
    const std::string piece = model.tokenizer().DecodeTokenCopy(
        static_cast<tokenization::QwenTokenizer::TokenId>(token));
    if (reply != nullptr) {
      reply->append(piece);
    }
    std::cout << piece << std::flush;
    // The token just drawn is the next row: one forward per token, the cache
    // already holding every position before it.
    session.Append(std::span<const q35::TokenId>(&token, 1));
  }
  std::cout << '\n';
  if (opt.verbose && generated > 0) {
    const double seconds = Elapsed(decode_start);
    std::cout << "Prefill " << prompt.size() << " tokens in "
              << prefill_seconds << " s ("
              << static_cast<double>(prompt.size()) / prefill_seconds
              << " tok/s)\nGenerated " << generated << " tokens in " << seconds
              << " s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
  }
  return 0;
}

[[nodiscard]] std::vector<tokenization::ChatMessage> Messages(
    const PromptOptions& opt) {
  std::vector<tokenization::ChatMessage> messages;
  if (!opt.system_prompt.empty()) {
    messages.emplace_back(tokenization::ChatRole::kSystem, opt.system_prompt);
  }
  return messages;
}

[[nodiscard]] q35::ChatOptions ResolveOptions(const PromptOptions& opt) {
  q35::ChatOptions options;
  const ReasoningOptions reasoning = PromptReasoningOptions(opt);
  options.enable_thinking = reasoning.enabled.value_or(true);
  return options;
}

[[nodiscard]] std::vector<q35::TokenId> Render(
    const q35::Model& model,
    std::span<const tokenization::ChatMessage> messages,
    const PromptOptions& opt) {
  std::string error;
  std::vector<tokenization::ChatTool> no_tools;
  const auto rendered = q35::ChatTemplate::Render(
      messages, no_tools, ResolveOptions(opt), model.template_variant(),
      &error);
  if (!rendered.has_value()) {
    throw std::invalid_argument("qwen35moe chat template: " + error);
  }
  return TokenizePrompt(model, rendered->text);
}

}  // namespace

bool IsQwen35Moe(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "qwen35moe";
}

int RunQwen35MoePrompt(const PromptOptions& opt,
                       std::chrono::steady_clock::time_point load_start) {
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cerr << "Ornith-1.5 session creation failed: " << error << '\n';
    return 1;
  }
  try {
    std::vector<q35::TokenId> prompt;
    if (opt.use_chat_template) {
      auto messages = Messages(opt);
      messages.emplace_back(tokenization::ChatRole::kUser, opt.prompt_text);
      prompt = Render(*model, messages, opt);
    } else {
      prompt = TokenizePrompt(*model, opt.prompt_text);
    }
    return Generate(opt, *model, *session, prompt, nullptr);
  } catch (const std::exception& e) {
    std::cerr << "Ornith-1.5 prompt failed: " << e.what() << '\n';
    return 1;
  }
}

int RunQwen35MoeChat(const PromptOptions& opt,
                     std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Ornith-1.5 interactive chat requires chat framing; use "
                 "prompt --raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cerr << "Ornith-1.5 session creation failed: " << error << '\n';
    return 1;
  }
  auto history = Messages(opt);
  std::cout << "=== Gufo Interactive Chat (Ornith-1.5) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" ||
        input == "quit") {
      break;
    }
    if (input.empty()) {
      continue;
    }
    history.emplace_back(tokenization::ChatRole::kUser, input);
    std::vector<q35::TokenId> prompt;
    try {
      prompt = Render(*model, history, opt);
    } catch (const std::exception& e) {
      std::cerr << "Ornith-1.5 chat template failed: " << e.what() << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (Generate(opt, *model, *session, prompt, &reply) != 0) {
      return 1;
    }
    // Every turn re-prefills the whole conversation: the cache is the shape of
    // the conversation, and the thinking the template drops from history is
    // not in it either.
    history.emplace_back(tokenization::ChatRole::kAssistant, std::move(reply));
    std::cout << '\n';
  }
  return 0;
}

}  // namespace gufo::cli
